import os
import platform
import psutil
from sciml_bench.core.system import HostSpec, DeviceSpec, DeviceSpecs

def test_host_spec():
    spec = HostSpec()

    assert isinstance(spec.disk_io, dict)
    assert isinstance(spec.net_io, dict)
    assert spec.system == platform.system()
    assert spec.name == os.name
    assert spec.release == platform.release()
    assert spec.num_cores == psutil.cpu_count()
    assert spec.total_memory == psutil.virtual_memory().total

def test_device_spec():
    spec = DeviceSpec(0)

    assert spec.is_multigpu_board == False

def test_device_specs():
    spec = DeviceSpecs()

    assert spec.device_count == 1
