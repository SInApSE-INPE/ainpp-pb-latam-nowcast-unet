# Scheduling with systemd

A systemd timer handles scheduling separately from processing. The service runs
one poll with `--scheduled` and exits. The timer polls every five minutes,
whether or not new data were found. No open terminal or background Python loop
is required.

This model consumes an **hourly** sequence: only `HH00` files are selected.
GSMaP NOW also has subhourly updates, but `HH30` is outside this pipeline's
input contract. Polling every five minutes reduces the delay after a new hourly
issue appears; it does not assume an exact publication minute. Detection occurs
at the first successful poll after publication. The provider does not send an
instant notification.

## What happens on each poll

1. Acquire an exclusive lock for the product and output directory. A concurrent
   scheduled poll reports `BUSY` and does not start another run.
2. List the FTP directory and identify the latest hour. If that issue already
   has a complete, verified output, neither weights nor input files are loaded.
3. Enforce freshness with `--max-age-hours`, pin the discovered hour, and
   request the complete window required by the checkpoint.
4. If hours are missing, log `WAITING` and exit normally; the timer retries
   later. Network and download failures use the `ftp` settings in
   `config/latam.json`: three attempts per operation, a 60-second timeout, and
   exponential delays starting at 30 seconds and capped at 120 seconds. Three
   attempts produce delays of 30 and 60 seconds.
5. A truncated gzip, invalid CRC, or wrong binary size also triggers a new
   transfer attempt. Only validated files enter the cache.
6. If all attempts fail, the service exits with an error recorded in the
   journal. The timer retries on its next activation; there is no competing
   retry scheduler.
7. Write the GeoTIFFs and publish the complete manifest. Later polls report
   `UP_TO_DATE`; completed issues are neither reprocessed nor overwritten.

The service prioritizes the **latest available issue**. After a long outage it
does not automatically backfill every intermediate issue. The persistent timer
resumes polling when the session or server starts, but does not build a
historical forecast queue. Use `--run-time` outside scheduled mode to
reprocess an older issue. Changes to data under an existing timestamp do not
trigger a new forecast.

## Input retention

FTP downloads live in the system temporary directory, isolated by user and
model (`nowcasting-<UID>/<model>`). Default retention is 24 hours from
download. Every service invocation removes expired files, even when the
forecast is already complete. Inputs supplied through `--input-dir` are never
deleted. Use `--cache-ttl-hours` to change retention or `--cache-dir` to
select another managed cache. The next manual FTP run also performs cleanup.
There is no resident cleanup process while the timer is disabled. Existing
files under `data/raw/` are not removed automatically.

## Prepare

Run from the `04_UNet` directory:

```bash
uv sync --locked
cp -n deploy/ftp.env.example .env
chmod 600 .env
${EDITOR:-vi} .env
```

Enter the username and password in `.env` as `KEY=value` lines, without
`export`. Git ignores this file; systemd reads it through `EnvironmentFile`.
Manual CLI runs still read environment variables. Do not put credentials on the
service command line. Quote passwords containing spaces according to
systemd's `EnvironmentFile` syntax.

Generate units containing this installation's absolute path:

```bash
.venv/bin/python deploy/render_units.py
systemd-analyze verify deploy/generated/unet-now.service deploy/generated/unet-now.timer
```

The generator **only writes to `deploy/generated/`**. It does not modify
system services or enable the timer. Regenerate and reinstall the units after
moving the project. The installation must be accessible from the service host
on a filesystem that supports `flock`.

## Install and enable for your user

```bash
mkdir -p "$HOME/.config/systemd/user"
cp deploy/generated/unet-now.service deploy/generated/unet-now.timer "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now unet-now.timer
systemctl --user start unet-now.service
```

The last command polls immediately; the timer handles later polls. Runtime
arguments use the pipeline CLI defaults without duplicating the horizon,
device, or directories in the unit. The service uses the installed `.venv`;
it does not run `uv sync` on every poll.

For execution after logout and at boot, the user service manager needs
**linger**:

```bash
loginctl show-user "$USER" -p Linger
```

If it reports `Linger=no`, ask an administrator to enable it:

```bash
sudo loginctl enable-linger "$USER"
```

If `systemctl --user` cannot connect to the bus, run the commands from a
normal user session on the operational host. On a cluster, ask the
administrator which service node permits persistent processes. These units
were prepared here, but were **not installed or enabled**.

## Monitor and adjust

```bash
systemctl --user list-timers unet-now.timer
systemctl --user status unet-now.service
journalctl --user -u unet-now.service -f
```

`inactive (dead)` is normal for a successful `oneshot` service; the timer
should remain active. Outputs use
`outputs/unet/gsmap_now/YYYY/MM/DD/HHMMZ/`.

- Poll frequency: edit only `OnCalendar` in `deploy/unet-now.timer`. The
  supplied value is `*-*-* *:0/5:00 UTC`. Regenerate and copy the units, then
  run `systemctl --user daemon-reload` and
  `systemctl --user restart unet-now.timer`.
- Retry policy: edit only the `ftp` section of `config/latam.json`, which is
  reloaded on every poll. Older configurations must include this section.
- Horizon, device, and directories: add the needed arguments to the command
  generated by the script, then regenerate the unit. Use another
  `--output-dir` for a different weight version or scientific configuration;
  completed issues in one directory are preserved.
- Execution limit: `TimeoutStartSec=20min` bounds a stuck poll. The timer
  does not restart a service that is still running.
- Stop new polls: `systemctl --user disable --now unet-now.timer`. To stop
  a run already in progress, use
  `systemctl --user stop unet-now.service`.

The lock protects **scheduled** invocations with the same directory and
product. Choose a different output directory for a parallel manual run. An
interrupted run does not publish a partial issue; forced termination may leave
hidden temporary directories that do not count as complete forecasts.

References: [systemd timers](https://www.freedesktop.org/software/systemd/man/252/systemd.timer.html)
and [GSMaP NOW format](https://sharaku.eorc.jaxa.jp/GSMaP/document/new/DataFormatDescription_NOW.pdf).
