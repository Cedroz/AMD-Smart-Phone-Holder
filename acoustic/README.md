# CPU Speaker Identification

Live 16 kHz microphone speaker identification using pretrained Resemblyzer embeddings. No model training is performed. The embedding call is isolated in `embed.py` so it can later be replaced by SpeechBrain ECAPA. Audio is processed on CPU.

## Environment

From this directory, create and activate the separate Python 3.11 environment, install CPU-only PyTorch and PortAudio, then install the pinned Python dependencies:

```bash
conda create -n speakerid python=3.11 -y
conda activate speakerid
conda install pytorch=2.5.1 cpuonly -c pytorch -c conda-forge -y
conda install portaudio -c conda-forge -y
python -m pip install -r requirements.txt
```

If pip reports a TLS/CA certificate verification error, point it at the machine's trusted CA bundle and retry:

```bash
export PIP_CERT=/etc/ssl/certs/ca-certificates.crt
python -m pip install -r requirements.txt
```

Use the CA bundle supplied by your operating system or organization if it is at a different path. Do not disable TLS verification.

## Mac Microphone Over SSH

The Python environment and speaker model run on the Linux SSH host, but that host cannot open the Mac's CoreAudio device directly. The live scripts therefore listen on `127.0.0.1:8765` by default. A small Mac-side sender captures mono 16 kHz audio and writes it through an SSH local-port tunnel; the audio transport is encrypted by SSH.

Run the following setup in the Mac's own **Terminal app**, not in the VS Code Remote-SSH terminal. First verify that `uname -s` prints `Darwin`. If it prints `Linux`, you are still running commands on the SSH host; stop there and open Terminal on the Mac. The Mac mic sender has to run on macOS to access the MacBook microphone.

Install PortAudio on the Mac, then make a small client folder and copy the two helper files from your SSH host:

```bash
uname -s
brew install portaudio
mkdir -p ~/speakerid-mic
scp aup@169.233.72.8:/home/aup/AMD-Smart-Phone-Holder/acoustic/config.py aup@169.233.72.8:/home/aup/AMD-Smart-Phone-Holder/acoustic/mac_mic_sender.py ~/speakerid-mic/
cd ~/speakerid-mic
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
 python3 -m pip install sounddevice==0.5.1 numpy==1.26.4
python3 mac_mic_sender.py --list-devices
python3 mac_mic_sender.py --device INDEX --test-seconds 5 --test-output test.wav
```

Choose the MacBook built-in microphone's index from the list. The test WAV is recorded on the Mac and is not sent to the remote host. If `brew` is not installed, install Homebrew on the Mac first; do not install Linux `python3-venv` or Linux PortAudio packages on the SSH server for this Mac-microphone setup.

For live use, open two terminals on the Mac and one terminal in the SSH workspace:

1. In the SSH workspace, start the receiver/inference process (or VAD / enrollment recorder):

   ```bash
   conda activate speakerid
   cd /home/aup/AMD-Smart-Phone-Holder/acoustic
   python live.py
   ```

2. In one Mac terminal, forward a local port to the SSH host's loopback receiver:

   ```bash
   ssh -N -L 18765:127.0.0.1:8765 aup@169.233.72.8
   ```

3. In the other Mac terminal, start the sender using the built-in mic index chosen above:

   ```bash
   cd ~/speakerid-mic
   source .venv/bin/activate
   python3 mac_mic_sender.py --device INDEX
   ```

The remote terminal prints when the sender connects. Stop the sender with Ctrl+C, then close the tunnel with Ctrl+C. `--local-mic` on remote scripts opts back into audio devices attached to the Linux host.

## Run Order

1. For a microphone physically attached to Linux, list its input devices and record a five-second test clip with `list_devices.py`. For the MacBook mic, use `mac_mic_sender.py --list-devices` and its `--test-seconds` option above instead:

   ```bash
   python list_devices.py
   python list_devices.py --device 4
   ```

   The local-host recorder writes `test.wav` in this directory. If there is no input, it exits with a clear message.

2. Record your own voice. You can record on the Mac **without an SSH tunnel**. In the Mac terminal, run the sender's local enrollment mode; it records three prompted 25-second clips into `~/speakerid-mic/enroll/Arnav/`:

   ```bash
   cd ~/speakerid-mic
   source .venv/bin/activate
   python mac_mic_sender.py --device 1 --enroll-speaker Arnav
   ```

   Then, in a separate SSH terminal, copy the WAV files to Linux and build the voiceprints:

   ```bash
   ssh aup@169.233.72.8 'mkdir -p /home/aup/AMD-Smart-Phone-Holder/acoustic/enroll/Arnav'
   scp ~/speakerid-mic/enroll/Arnav/*.wav aup@169.233.72.8:/home/aup/AMD-Smart-Phone-Holder/acoustic/enroll/Arnav/
   cd /home/aup/AMD-Smart-Phone-Holder/acoustic
   /home/aup/miniforge3/envs/speakerid/bin/python build_voiceprints.py
   ```

   The three clips are saved locally on the Mac first; the `scp` step transfers them to the Linux project. The older tunnel-based recorder is also available if you prefer it:

   ```bash
   /home/aup/miniforge3/envs/speakerid/bin/python record_enrollment.py --speaker Arnav
   ```

3. For plumbing tests before the other people can record, stream a small LibriSpeech validation subset into placeholder enrollment and labeled test folders for Edwin, Jaisree, and Matthew:

   ```bash
   python download_standins.py
   ```

   This requires network access. It downloads audio on demand through Hugging Face `datasets`. Placeholders are not the real people; delete each `enroll/<name>/standin_librispeech/` directory before recording that person's real enrollment.

4. Build one normalized voiceprint per speaker with available enrollment audio. Missing speakers are skipped, so self-versus-unknown testing can start with only your voice enrolled:

   ```bash
   python build_voiceprints.py
   ```

5. Start the Mac microphone sender/tunnel as described above. In the SSH workspace, check live VAD independently, then run speaker identification:

   ```bash
   python vad_live.py
   python live.py
   ```

   `live.speaker_stream()` is an importable generator yielding `(speaker, confidence, timestamp)` once per hop. Press Ctrl+C to stop either live script.

6. Evaluate labeled WAV files. Put test clips in directories named for the configured speaker, for example `test_wavs/Arnav/clip1.wav`. The evaluator applies the same 1.5-second windows, 0.5-second hops, VAD, embedding, unknown threshold, and smoothing used live. It reports window accuracy and a confusion matrix. For a concatenated turn-taking WAV with configured speakers in config order, provide the duration of each turn:

   ```bash
   python evaluate.py test_wavs
   python evaluate.py test_wavs --turn-taking-wav test_wavs/turn_taking.wav --turn-seconds 15
   ```

7. Preview motor/noise sensitivity using a labeled test folder plus a noise-only WAV:

   ```bash
   python noise_robustness.py test_wavs motor_noise.wav
   ```

8. Measure CPU embedding cost on this machine:

   ```bash
   python embed.py
   ```

## Configuration And Interpretation

Edit `config.py` to change names/order, microphone input mode and ports, device index, rate, window/hop, cosine unknown threshold, VAD aggressiveness, and smoothing count. The default threshold is a starting point, not a calibrated guarantee; use your own labeled clips with `evaluate.py` to tune it. Similarity confidence is cosine similarity, not a probability. With one enrolled person, other speech may still resemble that person enough to cross the threshold, so unknown rejection must be checked empirically.

## Unit Tests

```bash
python -m unittest discover -s tests -v
```