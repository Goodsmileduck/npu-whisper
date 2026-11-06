"""A tray icon for Linux desktops, over the session D-Bus.

pystray's Linux backends are an X11 tray icon with no menu (shown on KDE
through xembedsniproxy) or AppIndicator, which needs PyGObject built against
system headers. This speaks the protocols themselves: StatusNotifierItem for
the icon (KDE Plasma, GNOME with the AppIndicator extension, XFCE, LXQt...)
and com.canonical.dbusmenu for its menu. Its interface is the part of
pystray.Icon that ui/tray.py uses, and menus are pystray.Menu objects.
"""
import os
import threading

ITEM_PATH = "/StatusNotifierItem"
MENU_PATH = "/MenuBar"
ITEM_IFACE = "org.kde.StatusNotifierItem"
MENU_IFACE = "com.canonical.dbusmenu"
PROPS_IFACE = "org.freedesktop.DBus.Properties"
WATCHER = ("org.kde.StatusNotifierWatcher", "/StatusNotifierWatcher",
           "org.kde.StatusNotifierWatcher")


def watcher_available() -> bool:
    """Whether a StatusNotifierWatcher (the desktop's tray) is running."""
    try:
        from jeepney import DBusAddress, MessageType, new_method_call
        from jeepney.io.blocking import open_dbus_connection
        bus = DBusAddress("/org/freedesktop/DBus", bus_name="org.freedesktop.DBus",
                          interface="org.freedesktop.DBus")
        with open_dbus_connection(bus="SESSION") as conn:
            reply = conn.send_and_get_reply(
                new_method_call(bus, "NameHasOwner", "s", (WATCHER[0],)), timeout=2)
        return reply.header.message_type == MessageType.method_return and reply.body[0]
    except Exception:
        return False


def pixmaps(image):
    """PIL image -> SNI pixmaps: [(w, h, ARGB32 bytes in network order)]."""
    import numpy as np
    out = []
    for size in sorted({image.width, 22, 32, 64}):
        frame = image.convert("RGBA")
        if frame.width != size:
            frame = frame.resize((size, size))
        rgba = np.asarray(frame, dtype=np.uint8)
        out.append((size, size, rgba[..., [3, 0, 1, 2]].tobytes()))
    return out


class SniIcon:
    """pystray.Icon's subset: icon, title, menu, update_menu(), run(), stop()."""

    def __init__(self, name, icon=None, title="", menu=None):
        self.name = name
        self._icon, self._title, self._menu = icon, title, menu
        self.visible = False
        self._conn = None
        self._send_lock = threading.Lock()
        self._stopped = threading.Event()
        self._revision = 1
        self._items = {}  # dbusmenu id -> pystray.MenuItem, from the last layout
        self._bus_name = f"org.kde.StatusNotifierItem-{os.getpid()}-1"

    # -- the pystray interface -------------------------------------------

    @property
    def icon(self):
        return self._icon

    @icon.setter
    def icon(self, image):
        self._icon = image
        self._signal(ITEM_PATH, ITEM_IFACE, "NewIcon")

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, text):
        self._title = text
        self._signal(ITEM_PATH, ITEM_IFACE, "NewTitle")
        self._signal(ITEM_PATH, ITEM_IFACE, "NewToolTip")

    @property
    def menu(self):
        return self._menu

    @menu.setter
    def menu(self, menu):
        self._menu = menu

    def update_menu(self):
        self._revision += 1
        self._signal(MENU_PATH, MENU_IFACE, "LayoutUpdated", "ui", (self._revision, 0))

    def run(self, setup=None):
        """Register with the tray and answer it until stop()."""
        from jeepney.bus_messages import message_bus
        from jeepney.io.blocking import open_dbus_connection
        self._conn = open_dbus_connection(bus="SESSION")
        self._conn.send_and_get_reply(message_bus.RequestName(self._bus_name))
        self._call_watcher()
        self.visible = True
        if setup is not None:
            threading.Thread(target=setup, args=(self,), daemon=True).start()
        while not self._stopped.is_set():
            try:
                msg = self._conn.receive(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:
                break  # connection closed by stop()
            try:
                self._dispatch(msg)
            except Exception:
                pass  # one bad request must not take the icon down

    def stop(self):
        self._stopped.set()
        self.visible = False
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # -- D-Bus -------------------------------------------------------------

    def _call_watcher(self):
        from jeepney import DBusAddress, new_method_call
        name, path, iface = WATCHER
        address = DBusAddress(path, bus_name=name, interface=iface)
        with self._send_lock:
            self._conn.send(new_method_call(address, "RegisterStatusNotifierItem", "s",
                                            (self._bus_name,)))

    def _signal(self, path, iface, member, signature=None, body=()):
        if self._conn is None:
            return
        from jeepney import DBusAddress, new_signal
        try:
            with self._send_lock:
                self._conn.send(new_signal(DBusAddress(path, interface=iface), member,
                                           signature, body))
        except Exception:
            pass

    def _reply(self, msg, signature=None, body=()):
        from jeepney import new_method_return
        with self._send_lock:
            self._conn.send(new_method_return(msg, signature, body))

    def _error(self, msg, name="org.freedesktop.DBus.Error.UnknownMethod"):
        from jeepney import new_error
        with self._send_lock:
            self._conn.send(new_error(msg, name, "s", ("not supported",)))

    def _dispatch(self, msg):
        from jeepney import HeaderFields, MessageType
        header = msg.header
        if header.message_type != MessageType.method_call:
            return
        path = header.fields.get(HeaderFields.path)
        iface = header.fields.get(HeaderFields.interface)
        member = header.fields.get(HeaderFields.member)
        if iface == PROPS_IFACE:
            return self._properties(msg, member, path)
        if iface == "org.freedesktop.DBus.Introspectable":
            return self._reply(msg, "s", (_INTROSPECTION.get(path, _NODE),))
        if iface == "org.freedesktop.DBus.Peer" and member == "Ping":
            return self._reply(msg)
        if path == ITEM_PATH:
            if member in ("Activate", "SecondaryActivate"):
                self._activate_default()
                return self._reply(msg)
            if member in ("ContextMenu", "Scroll"):
                return self._reply(msg)
        if path == MENU_PATH:
            return self._menu_call(msg, member)
        self._error(msg)

    # -- StatusNotifierItem ------------------------------------------------

    def _item_properties(self):
        icon = pixmaps(self._icon) if self._icon is not None else []
        return {
            "Category": ("s", "ApplicationStatus"),
            "Id": ("s", self.name),
            "Title": ("s", self._title or self.name),
            "Status": ("s", "Active"),
            "WindowId": ("i", 0),
            "IconName": ("s", ""),
            "IconPixmap": ("a(iiay)", icon),
            "ToolTip": ("(sa(iiay)ss)", ("", [], self._title or self.name, "")),
            "ItemIsMenu": ("b", False),
            "Menu": ("o", MENU_PATH),
        }

    def _menu_properties(self):
        return {"Version": ("u", 3), "TextDirection": ("s", "ltr"),
                "Status": ("s", "normal"), "IconThemePath": ("as", [])}

    def _properties(self, msg, member, path):
        props = self._item_properties() if path == ITEM_PATH else self._menu_properties()
        if member == "GetAll":
            return self._reply(msg, "a{sv}", (props,))
        if member == "Get":
            _iface, name = msg.body
            if name in props:
                return self._reply(msg, "v", (props[name],))
            return self._error(msg, "org.freedesktop.DBus.Error.UnknownProperty")
        self._error(msg)

    def _activate_default(self):
        for item in self._menu.items if self._menu is not None else ():
            if getattr(item, "default", False) and item.enabled:
                self._run_item(item)
                return

    def _run_item(self, item):
        threading.Thread(target=item, args=(self,), daemon=True).start()

    # -- dbusmenu ------------------------------------------------------------

    def _layout(self):
        """The menu as dbusmenu's (id, properties, children) tree; id 0 is
        the root. Ids follow the item order of the current layout."""
        self._items = {}
        children = []
        # Iterating a pystray.Menu drops separators at either end.
        items = list(self._menu) if self._menu is not None else []
        for index, item in enumerate(items, start=1):
            if not item.visible:
                continue
            self._items[index] = item
            children.append(("(ia{sv}av)", (index, self._item_props(item), [])))
        return (0, {"children-display": ("s", "submenu")}, children)

    @staticmethod
    def _item_props(item):
        import pystray
        if item is pystray.Menu.SEPARATOR:
            return {"type": ("s", "separator")}
        props = {"label": ("s", str(item.text).replace("_", "__")),
                 "enabled": ("b", bool(item.enabled))}
        checked = item.checked
        if checked is not None:
            props["toggle-type"] = ("s", "radio" if getattr(item, "radio", False) else "checkmark")
            props["toggle-state"] = ("i", 1 if checked else 0)
        return props

    def _menu_call(self, msg, member):
        if member == "GetLayout":
            return self._reply(msg, "u(ia{sv}av)", (self._revision, self._layout()))
        if member == "GetGroupProperties":
            ids, _names = msg.body
            if not self._items:
                self._layout()
            rows = [(i, self._item_props(self._items[i])) for i in ids if i in self._items]
            return self._reply(msg, "a(ia{sv})", (rows,))
        if member == "GetProperty":
            item_id, name = msg.body
            props = self._item_props(self._items[item_id]) if item_id in self._items else {}
            if name in props:
                return self._reply(msg, "v", (props[name],))
            return self._error(msg, "org.freedesktop.DBus.Error.InvalidArgs")
        if member == "Event":
            item_id, event, _data, _timestamp = msg.body
            self._event(item_id, event)
            return self._reply(msg)
        if member == "EventGroup":
            for item_id, event, _data, _timestamp in msg.body[0]:
                self._event(item_id, event)
            return self._reply(msg, "ai", ([],))
        if member == "AboutToShow":
            return self._reply(msg, "b", (False,))
        if member == "AboutToShowGroup":
            return self._reply(msg, "aiai", ([], []))
        self._error(msg)

    def _event(self, item_id, event):
        item = self._items.get(item_id)
        if event == "clicked" and item is not None and item.enabled:
            self._run_item(item)


_NODE = '<node><interface name="org.freedesktop.DBus.Properties"/></node>'
_INTROSPECTION = {
    ITEM_PATH: f'<node><interface name="{ITEM_IFACE}"/><interface name="{PROPS_IFACE}"/></node>',
    MENU_PATH: f'<node><interface name="{MENU_IFACE}"/><interface name="{PROPS_IFACE}"/></node>',
}
