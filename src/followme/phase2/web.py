"""Camera-only diagnostic/calibration UI; no robot or movement endpoints."""
import asyncio
from pathlib import Path


def app_for(runtime, role):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, StreamingResponse
    app=FastAPI(title='FollowMe Phase 2 — camera only')

    @app.get('/',response_class=HTMLResponse)
    def index():
        return Path(__file__).with_name('console.html').read_text().replace('__ROLE__',role)

    @app.get('/api/user-state')
    def status(): return runtime.status()

    def local_origin(request):
        origin=request.headers.get('origin')
        expected=str(request.base_url).rstrip('/')
        if origin and origin.rstrip('/')!=expected:
            raise HTTPException(403,'Cross-origin controls are rejected')
        if request.headers.get('sec-fetch-site')=='cross-site':
            raise HTTPException(403,'Cross-site controls are rejected')

    @app.post('/api/reacquire')
    async def reacquire(request:Request):
        local_origin(request)
        try: runtime.reacquire()
        except ValueError as exc: raise HTTPException(409,str(exc))
        return runtime.status()

    @app.post('/api/zones')
    async def zones(request:Request):
        local_origin(request)
        if role!='external': raise HTTPException(403,'Use the external calibration console')
        raw=await request.body()
        if len(raw)>8192: raise HTTPException(413,'Calibration too large')
        try:
            body=await request.json()
            return runtime.save_zones(body['regions'],body.get('margin',.03))
        except (ValueError,KeyError,TypeError) as exc:
            raise HTTPException(422,str(exc))

    @app.get('/stream')
    async def stream():
        if runtime.preview(role) is None: raise HTTPException(503,'Camera unavailable or stale')
        async def frames():
            sequence=-1
            while not runtime.stopped.is_set():
                frame=runtime.preview(role)
                if frame is None: return
                if frame[0]!=sequence:
                    sequence,jpeg=frame
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg+b'\r\n'
                await asyncio.sleep(.1)
        return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store'})
    return app
