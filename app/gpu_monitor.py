"""GPU telemetry monitoring using pynvml."""

import asyncio
import logging
from typing import Any, Callable, Coroutine

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitors NVIDIA GPU telemetry via NVML and broadcasts over WebSocket."""

    def __init__(self) -> None:
        self._handle = None
        self._initialized = False
        self._task: asyncio.Task | None = None
        self._init_nvml()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_nvml(self) -> None:
        if not PYNVML_AVAILABLE:
            logger.warning("pynvml not installed – GPU telemetry disabled")
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._initialized = True
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode()
            logger.info("NVML initialised – GPU 0: %s", name)
        except pynvml.NVMLError as exc:
            logger.warning("NVML init failed: %s", exc)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def get_telemetry(self) -> dict[str, Any]:
        """Return a dict of GPU metrics. Returns zeros on error."""
        zeros: dict[str, Any] = {
            "gpu_util": 0,
            "mem_used_mb": 0,
            "mem_total_mb": 0,
            "mem_util": 0,
            "power_w": 0.0,
            "power_cap_w": 0.0,
            "temp_c": 0,
            "fan_speed": 0,
            "clock_sm_mhz": 0,
            "clock_mem_mhz": 0,
            "error": False,
        }
        if not self._initialized:
            zeros["error"] = True
            return zeros

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW → W
            try:
                power_cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000.0
            except pynvml.NVMLError:
                power_cap = 0.0
            temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(self._handle)
            except pynvml.NVMLError:
                fan = 0
            clock_sm = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
            clock_mem = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM)

            mem_used_mb = mem.used / (1024 ** 2)
            mem_total_mb = mem.total / (1024 ** 2)

            return {
                "gpu_util": util.gpu,
                "mem_used_mb": int(mem_used_mb),
                "mem_total_mb": int(mem_total_mb),
                "mem_util": util.memory,
                "power_w": round(power, 1),
                "power_cap_w": round(power_cap, 1),
                "temp_c": temp,
                "fan_speed": fan,
                "clock_sm_mhz": clock_sm,
                "clock_mem_mhz": clock_mem,
                "error": False,
            }
        except pynvml.NVMLError as exc:
            logger.warning("NVML read error: %s", exc)
            zeros["error"] = True
            return zeros

    def get_memory_info(self) -> dict[str, int]:
        """Return used / free / total VRAM in MB."""
        if not self._initialized:
            return {"used_mb": 0, "free_mb": 0, "total_mb": 0}
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {
                "used_mb": int(mem.used / (1024 ** 2)),
                "free_mb": int(mem.free / (1024 ** 2)),
                "total_mb": int(mem.total / (1024 ** 2)),
            }
        except pynvml.NVMLError:
            return {"used_mb": 0, "free_mb": 0, "total_mb": 0}

    # ------------------------------------------------------------------
    # Background telemetry broadcast
    # ------------------------------------------------------------------
    async def start_broadcast(
        self,
        broadcast_fn: Callable[[dict], Coroutine],
        interval: float = 0.5,
    ) -> None:
        """Start a background task that sends gpu_telemetry every *interval* seconds."""
        self.stop_broadcast()
        self._task = asyncio.create_task(self._broadcast_loop(broadcast_fn, interval))

    async def _broadcast_loop(
        self,
        broadcast_fn: Callable[[dict], Coroutine],
        interval: float,
    ) -> None:
        try:
            while True:
                telemetry = self.get_telemetry()
                telemetry.pop("error", None)
                msg = {"type": "gpu_telemetry", **telemetry}
                await broadcast_fn(msg)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    def stop_broadcast(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self.stop_broadcast()
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
            self._initialized = False
