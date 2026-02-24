export const onRequest = ({ request, env, next, waitUntil }: any) => {
    waitUntil(env.HITS_DB.prepare('INSERT INTO hits (path, ts, ua, country, ip) VALUES (?, ?, ?, ?, ?)')
        .bind(request.url, Date.now(), request.headers.get('user-agent'), request.cf?.country, request.headers.get('x-real-ip')).run())
    return next()
}
