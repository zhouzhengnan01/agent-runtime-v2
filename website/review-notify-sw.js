/* Review notifications Service Worker.
 *
 * Used by `website/review.html` to display system notifications via
 * `ServiceWorkerRegistration.showNotification()` and handle notification clicks.
 */

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("notificationclick", (event) => {
  const data = (event.notification && event.notification.data) || {};
  const url = typeof data.url === "string" && data.url ? data.url : "/review.html";
  event.notification?.close?.();

  event.waitUntil(
    (async () => {
      try {
        const clients = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
        const sameOrigin = [];
        for (const c of clients) {
          try {
            const u = new URL(c.url);
            if (u.origin === self.location.origin) sameOrigin.push(c);
          } catch {
            // ignore
          }
        }

        const candidates = sameOrigin.length ? sameOrigin : clients;
        for (const c of candidates) {
          try {
            if (String(c.url || "").includes("/review.html")) {
              await c.focus();
              if (typeof c.navigate === "function") await c.navigate(url);
              return;
            }
          } catch {
            // ignore
          }
        }

        for (const c of candidates) {
          try {
            await c.focus();
            if (typeof c.navigate === "function") await c.navigate(url);
            return;
          } catch {
            // ignore
          }
        }

        try {
          await self.clients.openWindow(url);
        } catch {
          // ignore
        }
      } catch {
        // ignore
      }
    })()
  );
});

