import os
import platform
import psutil
import socket
import cpuinfo
import pynvml as nv


def bytesto(bytes, to, bsize=1024):
    size = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    for i in range(size[to]):
        r = r / bsize
    return(r)

class HostSpec:

    def __init__(self, pid=None, per_device=False):
        self._per_device = per_device
        self._process = psutil.Process(pid)

    @property
    def name(self):
        return os.name

    @property
    def system(self):
        return platform.system()

    @property
    def node_name(self):
        return socket.gethostname()

    @property
    def ip_address(self):
        return socket.gethostbyname(self.node_name)

    @property
    def release(self):
        return platform.release()

    @property
    def num_cores(self):
        return psutil.cpu_count()

    @property
    def total_memory(self):
        mem = psutil.virtual_memory()
        return mem.total

    @property
    def cpu_percent(self):
        info = self._process.cpu_percent()
        return info

    @property
    def cpu_info(self):
        info = cpuinfo.get_cpu_info()
        keys = ['brand', 'arch', 'vendor_id', 'hz_advertised', 'hz_actual', 'model', 'family']
        return {key: value for key, value in info.items() if key in keys}

    @property
    def disk_io(self):
        info = self._process.io_counters()
        return info._asdict()

    @property
    def net_io(self):
        info = psutil.net_io_counters(pernic=self._per_device)
        if self._per_device:
            return {key: {k: v for k, v in value._asdict().items()} for key, value in info.items()}
        else:
            return {k: v for k, v in info._asdict().items()}

    @property
    def memory(self):
        memory_props = dict(psutil.virtual_memory()._asdict())

        metrics = {}
        metrics['free'] = bytesto(memory_props['memory_free'], 'm')
        metrics['used'] = bytesto(memory_props['memory_used'], 'm')
        metrics['available'] = bytesto(memory_props['memory_available'], 'm')
        metrics['utilization'] = memory_props['memory_percent']
        return metrics

class DeviceSpecs:

    def __init__(self):

        try:
            # We're on a machine with Nvidia libraries
            nv.nvmlInit()
            self._device_count = nv.nvmlDeviceGetCount()
            self._specs = [DeviceSpec(i) for i in range(self.device_count)]
        except nv.NVMLError_LibraryNotFound:
            # This device has no GPU or no Nvidia libraries are installed.
            self._device_count = 0
            self._specs = []

    @property
    def device_count(self):
        return self._device_count

    def get_device_info(self, index):
        spec = self.get_device_spec(index)

        info = {
            'uuid': spec.uuid,
            'name': spec.name,
            'brand': spec.brand,
            'minor_number': spec.minor_number,
            'multi_gpu_board': spec.is_multigpu_board,
            'memory': dict(total=spec.memory['total'])
        }

        return info

    def get_device_spec(self, index):
        if index >= self._device_count:
            raise RuntimeError('Cannot device index {} is greater than number of devices')

        return self._specs[index]

    def device_specs(self):
        return self._specs

    @property
    def uuids(self):
        return {'gpu_{}_uuid'.format(i): spec.uuid for i, spec in enumerate(self._specs)}

    @property
    def names(self):
        return {'gpu_{}_name'.format(i): spec.name for i, spec in enumerate(self._specs)}

    @property
    def brands(self):
        return {'gpu_{}_brand'.format(i): spec.brand for i, spec in enumerate(self._specs)}

    @property
    def minor_numbers(self):
        return {'gpu_{}_minor_number'.format(i): spec.minor_number for i, spec in enumerate(self._specs)}

    @property
    def is_multigpu_board(self):
        return {'gpu_{}_is_mulitgpu_board'.format(i): spec.is_multigpu_board for i, spec in enumerate(self._specs)}

    @property
    def power_usage(self):
        return {'gpu_{}_power'.format(i): spec.power_usage for i, spec in enumerate(self._specs)}

    @property
    def memory(self):
        memory_info = {}
        for i, spec in enumerate(self._specs):
            for key, value in spec.memory.items():
                memory_info['gpu_{}_memory_{}'.format(i, key)] = value
        return memory_info

    @property
    def utilization_rates(self):
        memory_info = {}
        for i, spec in enumerate(self._specs):
            for key, value in spec.utilization_rates.items():
                memory_info['gpu_{}_{}'.format(i, key)] = value
        return memory_info

class DeviceSpec:

    def __init__(self, index):
        try:
            nv.nvmlInit()
            self._handle = nv.nvmlDeviceGetHandleByIndex(index)
        except nv.NVMLError_LibraryNotFound:
            pass

    @property
    def uuid(self):
        """ NVIDIA device UUID """
        return nv.nvmlDeviceGetUUID(self._handle).decode()

    @property
    def name(self):
        """ NVIDIA device name """
        return nv.nvmlDeviceGetName(self._handle).decode()

    @property
    def brand(self):
        """ Device brand name as a string

        This function maps the device code to a string representation using the
        following enum:

            NVML_BRAND_UNKNOWN = 0
            NVML_BRAND_QUADRO = 1
            NVML_BRAND_TESLA = 2
            NVML_BRAND_NVS = 3
            NVML_BRAND_GRID = 4
            NVML_BRAND_GEFORCE = 5
            NVML_BRAND_TITAN = 6
        """
        brand_enum = nv.nvmlDeviceGetBrand(self._handle)

        if brand_enum == 1:
            return 'Quadro'
        elif brand_enum == 2:
            return 'Tesla'
        elif brand_enum == 3:
            return 'NVS'
        elif brand_enum == 4:
            return 'Grid'
        elif brand_enum == 5:
            return 'GeForce'
        elif brand_enum == 6:
            return 'Titan'
        else:
            return 'Unknown'

    @property
    def minor_number(self):
        return nv.nvmlDeviceGetMinorNumber(self._handle)

    @property
    def is_multigpu_board(self):
        return nv.nvmlDeviceGetMultiGpuBoard(self._handle)

    @property
    def utilization_rates(self):
        rates = nv.nvmlDeviceGetUtilizationRates(self._handle)
        return dict(gpu=rates.gpu, memory=rates.memory)

    @property
    def memory(self):
        """ Total, free, and used memory in bytes"""
        info = nv.nvmlDeviceGetMemoryInfo(self._handle)
        return dict(free=info.free, total=info.total, used=info.used)

    @property
    def power_usage(self):
        """ Power usage for the device in milliwatts

        From the NVIDIA documentation:
         - On Fermi and Kepler GPUs the reading is accurate to within +/- 5% of current power draw.
        """
        return nv.nvmlDeviceGetPowerUsage(self._handle)

