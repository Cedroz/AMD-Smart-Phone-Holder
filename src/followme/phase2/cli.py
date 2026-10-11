"""Independent Phase 2 commands; robot I/O is not imported or configured."""
import argparse
import json
from pathlib import Path
import signal
import threading
import time


def main():
    parser=argparse.ArgumentParser(description='Camera-only Phase 2 perception')
    parser.add_argument('command',choices=['perceive','calibrate-zones','evaluate-perception'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--host',default='0.0.0.0')
    parser.add_argument('--external-port',type=int,default=8010)
    parser.add_argument('--wrist-port',type=int,default=8011)
    parser.add_argument('--preview-sources',action='store_true',help='Use legacy HTTP previews for occupied cameras, native capture for unowned cameras; proxy timestamps reflect delivery')
    parser.add_argument('--max-seconds',type=float)
    parser.add_argument('--log-dir',type=Path)
    parser.add_argument('--log',type=Path)
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--inference-width',type=int,help='Session-only resize override; does not edit configuration')
    args=parser.parse_args()
    if args.command=='evaluate-perception':
        if not args.log: parser.error('--log is required')
        from followme.phase2.evaluation import replay
        result=replay(args.log,args.annotations)
        encoded=json.dumps(result,indent=2,allow_nan=False)
        if args.output: args.output.write_text(encoded+'\n')
        print(encoded)
        return 0 if result['replay_matches'] and result['live_acceptance']!='failed' else 1
    if args.external_port==args.wrist_port: parser.error('Preview ports must differ')
    if args.max_seconds is not None and args.max_seconds<=0: parser.error('--max-seconds must be positive')
    from followme.phase2.runtime import Runtime
    from followme.phase2.web import app_for
    import uvicorn
    override={'inference_width':args.inference_width} if args.inference_width else None
    runtime=Runtime(args.root,args.log_dir,args.preview_sources,override)
    stop=threading.Event()
    servers=[]
    threads=[]
    def shutdown(*_): stop.set()
    signal.signal(signal.SIGINT,shutdown)
    signal.signal(signal.SIGTERM,shutdown)
    try:
        runtime.start()
        for role,port in [('external',args.external_port),('wrist',args.wrist_port)]:
            server=uvicorn.Server(uvicorn.Config(app_for(runtime,role),host=args.host,port=port,log_level='warning'))
            servers.append(server)
            thread=threading.Thread(target=server.run,daemon=True)
            threads.append(thread);thread.start()
        print(f'Phase 2 camera-only consoles: http://{runtime.config["host"]}:{args.external_port}/ and :{args.wrist_port}/',flush=True)
        print(f'Evidence: {runtime.log_dir}',flush=True)
        start=time.monotonic()
        while not stop.wait(.2):
            if args.max_seconds and time.monotonic()-start>=args.max_seconds: break
            if any(not thread.is_alive() for thread in threads):
                raise RuntimeError('Diagnostic server failed; check port ownership')
    except BaseException as exc:
        runtime.record({'type':'session_error','error':f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        for server in servers: server.should_exit=True
        for thread in threads: thread.join(timeout=2)
        runtime.close()
    return 0


if __name__=='__main__': raise SystemExit(main())
