"""Phase 5 operator service; no SDK operations in request handlers."""
import asyncio
import secrets
import urllib.request


def app_for(runtime, config, token=None):
    from fastapi import FastAPI, HTTPException, Request, WebSocket
    from fastapi.responses import HTMLResponse, StreamingResponse, Response
    token = token or secrets.token_urlsafe(32)
    app = FastAPI(title='FollowMe Phase 5')
    app.state.control_token = token

    def authorize(request):
        if request.headers.get('origin') not in config['allowed_origins'] or request.headers.get('sec-fetch-site') == 'cross-site':
            raise HTTPException(403, 'origin_rejected')
        if not secrets.compare_digest(request.headers.get('authorization', ''), 'Bearer '+token):
            raise HTTPException(401, 'control_token_required')
        return request.headers.get('x-operator-id', '')

    @app.get('/', response_class=HTMLResponse)
    def index():
        # Token travels in headers, never URLs or logs. Console owns a random lease ID.
        return '''<!doctype html><meta name="viewport" content="width=device-width"><title>FollowMe controls</title>
<style>body{font:18px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}button{font:inherit;margin:.4rem;padding:.7rem}pre{white-space:pre-wrap}#stop{background:#a00;color:white}.cameras{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}.camera img{width:100%;min-height:180px;object-fit:contain;background:#18202b;border-radius:8px}.camera p{font-size:14px}h2{font-size:20px}</style>
<h1>FollowMe operator console</h1><p>Starts in Hold. Software Stop does not replace the physical stop.</p>
<p id="motion-status"></p><button id="follow" onclick="send('mode',{mode:'follow'})">Follow</button><button onclick="send('mode',{mode:'hold'})">Hold</button>
<button id="stop" onclick="send('stop',{})">Stop</button><button onclick="send('reset',{})">Reset to Hold</button>
<button onclick="send('reacquire',{})">Reacquire user in Hold</button>
<button id="calibrate" onclick="send('calibrate',{})">Calibrate direction (small jog)</button>
<span id="directions" hidden><button onclick="send('reverse',{axis:'horizontal'})">Reverse X direction</button><button onclick="send('reverse',{axis:'vertical'})">Reverse Y direction</button></span>
<p id="instructions">For direction calibration, stand still as the only visible person. The arm makes a small sideways movement. Then select Follow. Live trials remain within 2° of the starting pan angle.</p>
<p id="error"></p><p id="camera-note"></p><section class="cameras" aria-label="Camera tracking previews">
<article class="camera"><h2>Robot-mounted tracking camera</h2><img id="external-camera" alt="Robot-mounted tracking preview"><p id="external-camera-status" aria-live="polite">Connecting…</p></article>
</section><pre id="state"></pre><script>
const token=TOKEN, operator=crypto.randomUUID?.()||Array.from(crypto.getRandomValues(new Uint8Array(16)),x=>x.toString(16).padStart(2,'0')).join('');
async function send(path,body){try{const r=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token,'X-Operator-ID':operator},body:JSON.stringify(body)});const d=await r.json();document.getElementById('error').textContent=r.ok?'':JSON.stringify(d)}catch(e){document.getElementById('error').textContent=String(e)}}
async function poll(){try{const state=await(await fetch('/api/status')).json();document.getElementById('state').textContent=JSON.stringify(state,null,2);document.getElementById('follow').textContent=state.backend==='fake'?'Follow (simulation)':'Follow';document.getElementById('motion-status').textContent=state.backend==='fake'?'Robot motion disabled. '+(state.live_motion_blockers||[]).join('; '):'Physical robot connected.';if(state.direction_source==='track.py default signs'){document.getElementById('calibrate').hidden=true;document.getElementById('directions').hidden=false;document.getElementById('instructions').textContent='Select Follow to track continuously in X and Y using face/body detection. Direction buttons return to Hold; select Follow again after reversing. Local travel limits remain active.'}}catch(e){document.getElementById('error').textContent='Disconnected'}}
async function cameras(){try{const info=await(await fetch('/api/cameras')).json();document.getElementById('camera-note').textContent=info.note;
for(const role of ['external']){const img=document.getElementById(role+'-camera'),label=document.getElementById(role+'-camera-status');
let timer;
function connect(){clearTimeout(timer);img.src='/camera-frames/'+role+'?preview='+Date.now();timer=setTimeout(connect,10000)}
img.onload=()=>{clearTimeout(timer);label.textContent='Live camera preview';timer=setTimeout(connect,200)};
img.onerror=()=>{clearTimeout(timer);label.textContent='Camera unavailable or stale — retrying';timer=setTimeout(connect,1000)};connect();}
}catch(e){document.getElementById('camera-note').textContent='Camera previews unavailable'}}
send('heartbeat',{});setInterval(()=>{if(!document.hidden)send('heartbeat',{})},1000);setInterval(poll,200);poll();cameras();
</script>'''.replace('TOKEN', __import__('json').dumps(token))

    @app.get('/api/status')
    def status():
        return runtime.status()

    @app.get('/api/cameras')
    def camera_sources():
        owned = getattr(runtime, 'provider', None) is not None
        return {'note': ('Camera-backed tracking previews.' if owned else
                        'This run uses simulated tracking. Camera feeds below are viewing only.'),
                'roles': {role: ([{'path':f'/streams/{role}', 'label':'Phase 5 tracking overlay'}] if owned else []) +
                          [{'path':f'/camera-feeds/{role}', 'label':'Live camera preview'}] +
                          [{'port':port, 'path':'/stream', 'label':'Tracking diagnostic overlay'},
                           {'port':fallback, 'path':'/', 'label':'Camera-only feed'}]
                          for role, port, fallback in [('external',8010,8000)]}}

    @app.get('/camera-feeds/{role}')
    async def camera_feed(role: str):
        # Reuse camera owners over loopback; never open a camera or serial device.
        sources = {'external': [(8010, '/stream'), (8000, '/')],
                   'wrist': [(8011, '/stream'), (8001, '/')]}
        if role not in sources:
            raise HTTPException(404, 'unknown_camera')

        def connect():
            for port, path in sources[role]:
                upstream = None
                try:
                    upstream = urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=2)
                    content_type = upstream.headers.get('Content-Type', '')
                    if not content_type.startswith('multipart/x-mixed-replace'):
                        raise ValueError('invalid_camera_stream')
                    first = upstream.read1(65536)
                    if not first:
                        raise ValueError('empty_camera_stream')
                    return upstream, content_type, first
                except Exception:
                    if upstream is not None:
                        upstream.close()
            return None

        connected = await asyncio.to_thread(connect)
        if connected is None:
            raise HTTPException(503, 'camera_unavailable_or_stale')
        upstream, content_type, first = connected

        async def chunks():
            try:
                yield first
                while not runtime.stopped.is_set():
                    chunk = await asyncio.to_thread(upstream.read1, 65536)
                    if not chunk:
                        break
                    yield chunk
            except (OSError, TimeoutError):
                return
            finally:
                upstream.close()
        return StreamingResponse(chunks(), media_type=content_type,
                                 headers={'Cache-Control': 'no-store'})

    @app.get('/camera-frames/{role}')
    async def camera_frame(role: str):
        # Plain JPEG refreshes also work in browsers that reject MJPEG images.
        if role not in ('external', 'wrist'):
            raise HTTPException(404, 'unknown_camera')
        provider = getattr(runtime, 'provider', None)
        if provider is not None:
            frame = await asyncio.to_thread(provider.preview, role)
            if frame is not None:
                return Response(frame[1], media_type='image/jpeg',
                                headers={'Cache-Control': 'no-store'})
        stream = await camera_feed(role)
        data = bytearray()
        try:
            async for chunk in stream.body_iterator:
                data.extend(chunk)
                start = data.find(b'\xff\xd8')
                end = data.find(b'\xff\xd9', max(start, 0))
                if start >= 0 and end >= 0:
                    return Response(bytes(data[start:end+2]), media_type='image/jpeg',
                                    headers={'Cache-Control': 'no-store'})
                if len(data) > 2_000_000:
                    break
        finally:
            await stream.body_iterator.aclose()
        raise HTTPException(503, 'camera_frame_unavailable')

    @app.get('/streams/{role}')
    async def stream(role: str):
        if role not in ('external','wrist'):
            raise HTTPException(404,'unknown_camera')
        provider = getattr(runtime, 'provider', None)
        if provider is None:
            raise HTTPException(503,'camera_backed_perception_not_enabled')
        if await asyncio.to_thread(provider.preview, role) is None:
            raise HTTPException(503,'camera_unavailable_or_stale')
        async def frames():
            sequence = -1
            while not runtime.stopped.is_set():
                frame = await asyncio.to_thread(provider.preview, role)
                if frame is None:
                    return
                if frame[0] != sequence:
                    sequence, jpeg = frame
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg+b'\r\n'
                await asyncio.sleep(.1)
        return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame',
                                 headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})

    @app.post('/api/{action}')
    async def control(action: str, request: Request):
        client = authorize(request)
        try:
            if action == 'mode':
                body = await request.json()
                if not isinstance(body, dict) or set(body) != {'mode'}:
                    raise ValueError('mode_only')
                runtime.integration.set_mode(body['mode'], client)
            elif action == 'heartbeat':
                runtime.integration.heartbeat(client)
            elif action == 'stop':
                runtime.integration.stop()
            elif action == 'reset':
                runtime.integration.reset()
            elif action == 'reacquire':
                if runtime.integration.mode != 'hold':
                    raise ValueError('hold_required')
                provider = getattr(runtime, 'provider', None)
                if provider is None:
                    raise ValueError('real_perception_required')
                provider.reacquire()
            elif action == 'calibrate':
                if not hasattr(runtime.integration,'calibrate'):
                    raise ValueError('continuous_centering_required')
                runtime.integration.calibrate(client)
            elif action == 'reverse':
                body=await request.json()
                if not isinstance(body,dict) or set(body)!={'axis'}:
                    raise ValueError('axis_only')
                runtime.integration.reverse(body['axis'])
            else:
                raise HTTPException(404, 'unknown_control')
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(409, str(exc))
        return runtime.status()

    @app.websocket('/ws/state')
    async def telemetry(socket: WebSocket):
        protocols = socket.headers.get('sec-websocket-protocol', '').split(', ')
        if socket.headers.get('origin') not in config['allowed_origins'] or token not in protocols or 'followme' not in protocols:
            await socket.close(code=1008)
            return
        await socket.accept(subprotocol='followme')
        try:
            while True:
                await socket.send_json(runtime.status())
                await asyncio.sleep(.2)
        except Exception:
            return
    return app
