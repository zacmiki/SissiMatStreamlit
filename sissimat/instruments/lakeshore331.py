"""Read-only serial interface for the Lake Shore Model 331 controller."""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import serial
from serial.tools import list_ports


@dataclass(frozen=True)
class SerialPortInfo:
    """A serial port shown in the Streamlit port selector."""

    device: str
    description: str
    hardware_id: str

    @property
    def label(self) -> str:
        detail = self.description if self.description and self.description != "n/a" else "Serial port"
        return f"{self.device} — {detail}"


@dataclass(frozen=True)
class TemperatureReading:
    """A simultaneous pair of Model 331 input readings."""

    timestamp: datetime
    channel_a_kelvin: float
    channel_b_kelvin: float


def available_serial_ports() -> list[SerialPortInfo]:
    """Return all serial ports currently reported by the operating system."""

    return [
        SerialPortInfo(port.device, port.description, port.hwid)
        for port in sorted(list_ports.comports(), key=lambda item: item.device)
    ]


class LakeShore331:
    """Minimal read-only Lake Shore 331 RS-232 client.

    The controller uses 9600 baud, seven data bits, odd parity, one stop bit,
    CR/LF terminators, and software timing. Only query commands are exposed.
    """

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 9600,
        timeout: float = 2.0,
        query_delay: float = 0.06,
        serial_factory: Callable[..., serial.Serial] | None = None,
    ) -> None:
        factory = serial_factory or serial.Serial
        self.port = port
        self.baudrate = baudrate
        self.query_delay = query_delay
        self._lock = threading.Lock()
        self._serial = factory(
            port=port,
            baudrate=baudrate,
            bytesize=serial.SEVENBITS,
            parity=serial.PARITY_ODD,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

    @property
    def is_open(self) -> bool:
        return bool(getattr(self._serial, "is_open", True))

    def close(self) -> None:
        if self.is_open:
            self._serial.close()

    def query(self, command: str) -> str:
        """Send one query and return its response without terminators."""

        with self._lock:
            self._serial.reset_input_buffer()
            self._serial.write((command + "\r\n").encode("ascii"))
            self._serial.flush()
            response = self._serial.read_until(b"\n")
            time.sleep(self.query_delay)
        if not response:
            raise TimeoutError(f"No response to {command!r} on {self.port}")
        return response.decode("ascii", errors="replace").strip("\x00\r\n ")

    def identify(self) -> str:
        return self.query("*IDN?")

    def read_kelvin(self, channel: str) -> float:
        normalized = channel.strip().upper()
        if normalized not in {"A", "B"}:
            raise ValueError("Channel must be A or B")
        response = self.query(f"KRDG? {normalized}")
        try:
            return float(response)
        except ValueError as error:
            raise RuntimeError(
                f"Unexpected channel {normalized} response: {response!r}"
            ) from error

    def read_temperatures(self) -> TemperatureReading:
        return TemperatureReading(
            timestamp=datetime.now(timezone.utc),
            channel_a_kelvin=self.read_kelvin("A"),
            channel_b_kelvin=self.read_kelvin("B"),
        )

    def __enter__(self) -> "LakeShore331":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


SNAPSHOT_FIELDS = [
    "timestamp_utc",
    "controller",
    "serial_port",
    "channel_a_kelvin",
    "channel_b_kelvin",
    "note",
]


def append_temperature_snapshot(
    path: str | Path,
    reading: TemperatureReading,
    *,
    controller: str,
    serial_port: str,
    note: str = "",
) -> Path:
    """Append one user-requested temperature snapshot to a CSV file."""

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": reading.timestamp.astimezone(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "controller": controller,
                "serial_port": serial_port,
                "channel_a_kelvin": reading.channel_a_kelvin,
                "channel_b_kelvin": reading.channel_b_kelvin,
                "note": note.strip(),
            }
        )
    return output.resolve()
