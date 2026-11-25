"""Minimal synchronous Libretro host for NES cores (software video + PCM audio)."""
import ctypes as C
from pathlib import Path
import numpy as np

ENV_SET_PIXEL_FORMAT = 10
ENV_GET_SYSTEM_DIRECTORY = 9
ENV_GET_SAVE_DIRECTORY = 31
ENV_SET_VARIABLES = 16
ENV_GET_VARIABLE = 15
ENV_GET_VARIABLE_UPDATE = 17
ENV_SET_INPUT_DESCRIPTORS = 11
ENV_GET_LOG_INTERFACE = 27
ENV_GET_CAN_DUPE = 3
ENV_SET_SUPPORT_NO_GAME = 18
ENV_GET_CORE_OPTIONS_VERSION = 52
ENV_GET_LANGUAGE = 39
ENV_GET_INPUT_BITMASKS = 51
DEVICE_JOYPAD = 1
MEMORY_SAVE_RAM, MEMORY_RTC, MEMORY_SYSTEM_RAM, MEMORY_VIDEO_RAM = 0, 1, 2, 3
PIXEL_0RGB1555, PIXEL_XRGB8888, PIXEL_RGB565 = 0, 1, 2

class GameInfo(C.Structure):
    _fields_ = [('path', C.c_char_p), ('data', C.c_void_p), ('size', C.c_size_t), ('meta', C.c_char_p)]
class Geometry(C.Structure):
    _fields_ = [('base_width', C.c_uint), ('base_height', C.c_uint), ('max_width', C.c_uint), ('max_height', C.c_uint), ('aspect_ratio', C.c_float)]
class Timing(C.Structure):
    _fields_ = [('fps', C.c_double), ('sample_rate', C.c_double)]
class AVInfo(C.Structure):
    _fields_ = [('geometry', Geometry), ('timing', Timing)]
class Variable(C.Structure):
    _fields_ = [('key', C.c_char_p), ('value', C.c_char_p)]

ENV_CB = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
VIDEO_CB = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
AUDIO_CB = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
AUDIO_BATCH_CB = C.CFUNCTYPE(C.c_size_t, C.c_void_p, C.c_size_t)
INPUT_POLL_CB = C.CFUNCTYPE(None)
INPUT_STATE_CB = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)

class LibretroHost:
    def __init__(self, core, rom, system_dir=None, save_dir=None):
        self.core_path = str(Path(core).expanduser().resolve())
        self.rom_path = str(Path(rom).expanduser().resolve())
        if not Path(self.core_path).is_file() or not Path(self.rom_path).is_file():
            raise FileNotFoundError('Core or ROM not found')
        self.system_dir = str(Path(system_dir or Path(self.rom_path).parent).resolve()).encode()
        self.save_dir = str(Path(save_dir or Path(self.rom_path).parent).resolve()).encode()
        self.lib = C.CDLL(self.core_path)
        self.buttons = [set(), set()]
        self.frame = None
        self.audio = []
        self.pixel_format = PIXEL_0RGB1555
        self.options = {}
        self._buffers = []
        self._initialized = False
        self._loaded = False
        self._bind()
        self._callbacks()
        self.lib.retro_set_environment(self._env)
        self.lib.retro_set_video_refresh(self._video)
        self.lib.retro_set_audio_sample(self._audio)
        self.lib.retro_set_audio_sample_batch(self._audio_batch)
        self.lib.retro_set_input_poll(self._poll)
        self.lib.retro_set_input_state(self._input)
        try:
            self.lib.retro_init()
            self._initialized = True
            info = GameInfo(self.rom_path.encode(), None, 0, None)
            if not self.lib.retro_load_game(C.byref(info)):
                raise RuntimeError('retro_load_game failed: verify ROM and core')
            self._loaded = True
            self.av = AVInfo()
            self.lib.retro_get_system_av_info(C.byref(self.av))
            if not (1 <= self.av.timing.fps <= 1000 and 8000 <= self.av.timing.sample_rate <= 384000):
                raise RuntimeError('Invalid AV timing from core')
        except BaseException:
            self.close()
            raise

    def _bind(self):
        signatures = {
            'retro_set_environment': ([ENV_CB], None),
            'retro_set_video_refresh': ([VIDEO_CB], None),
            'retro_set_audio_sample': ([AUDIO_CB], None),
            'retro_set_audio_sample_batch': ([AUDIO_BATCH_CB], None),
            'retro_set_input_poll': ([INPUT_POLL_CB], None),
            'retro_set_input_state': ([INPUT_STATE_CB], None),
            'retro_init': ([], None), 'retro_deinit': ([], None),
            'retro_load_game': ([C.POINTER(GameInfo)], C.c_bool),
            'retro_unload_game': ([], None), 'retro_run': ([], None),
            'retro_get_system_av_info': ([C.POINTER(AVInfo)], None),
            'retro_reset': ([], None),
            'retro_get_memory_data': ([C.c_uint], C.c_void_p),
            'retro_get_memory_size': ([C.c_uint], C.c_size_t),
            'retro_serialize_size': ([], C.c_size_t),
            'retro_serialize': ([C.c_void_p, C.c_size_t], C.c_bool),
            'retro_unserialize': ([C.c_void_p, C.c_size_t], C.c_bool),
        }
        for name, (args, ret) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ret

    def _callbacks(self):
        @ENV_CB
        def env(cmd, data):
            if cmd == ENV_SET_PIXEL_FORMAT and data:
                fmt = C.cast(data, C.POINTER(C.c_int))[0]
                if fmt in (PIXEL_0RGB1555, PIXEL_XRGB8888, PIXEL_RGB565):
                    self.pixel_format = fmt
                    return True
                return False
            if cmd in (ENV_GET_SYSTEM_DIRECTORY, ENV_GET_SAVE_DIRECTORY) and data:
                pointer = C.cast(data, C.POINTER(C.c_char_p))
                pointer[0] = self.system_dir if cmd == ENV_GET_SYSTEM_DIRECTORY else self.save_dir
                return True
            if cmd == ENV_GET_CAN_DUPE and data:
                C.cast(data, C.POINTER(C.c_bool))[0] = False
                return True
            if cmd == ENV_GET_INPUT_BITMASKS:
                return False
            if cmd == ENV_GET_CORE_OPTIONS_VERSION and data:
                C.cast(data, C.POINTER(C.c_uint))[0] = 0
                return True
            if cmd == ENV_GET_LANGUAGE and data:
                C.cast(data, C.POINTER(C.c_uint))[0] = 0
                return True
            if cmd == ENV_SET_VARIABLES and data:
                ptr = C.cast(data, C.POINTER(Variable))
                i = 0
                while ptr[i].key and i < 1024:
                    key = ptr[i].key.decode()
                    val = (ptr[i].value or b'').decode()
                    choices = val.split(';', 1)[-1].strip().split('|')
                    self.options[key] = choices[0].strip()
                    i += 1
                return True
            if cmd == ENV_GET_VARIABLE and data:
                var = C.cast(data, C.POINTER(Variable))
                key = var[0].key.decode() if var[0].key else ''
                if key in self.options:
                    var[0].value = self.options[key].encode()
                    self._buffers.append(var[0].value)
                    return True
                return False
            if cmd == ENV_GET_VARIABLE_UPDATE and data:
                C.cast(data, C.POINTER(C.c_bool))[0] = False
                return True
            if cmd == ENV_SET_INPUT_DESCRIPTORS:
                return True
            return False

        @VIDEO_CB
        def video(data, width, height, pitch):
            if not data or width == 0 or height == 0:
                return
            if width > 4096 or height > 4096:
                return
            bpp = 4 if self.pixel_format == PIXEL_XRGB8888 else 2
            if pitch < width * bpp:
                return
            raw = C.string_at(data, pitch * height)
            self.frame = decode_frame(raw, width, height, pitch, self.pixel_format)

        @AUDIO_CB
        def audio(left, right):
            self.audio.append(np.array([[left, right]], dtype=np.int16))

        @AUDIO_BATCH_CB
        def audio_batch(data, frames):
            if data and frames:
                self.audio.append(np.ctypeslib.as_array(C.cast(data, C.POINTER(C.c_int16)), shape=(frames * 2,)).reshape(-1, 2).copy())
            return frames

        @INPUT_POLL_CB
        def poll():
            pass

        @INPUT_STATE_CB
        def input_state(port, device, index, button):
            if device != DEVICE_JOYPAD or index != 0 or port >= len(self.buttons):
                return 0
            return int(button in self.buttons[port])

        self._env, self._video, self._audio, self._audio_batch = env, video, audio, audio_batch
        self._poll, self._input = poll, input_state

    def step(self, p1=(), p2=()):
        self.buttons = [set(p1), set(p2)]
        self.audio.clear()
        self.lib.retro_run()
        samples = np.concatenate(self.audio, axis=0) if self.audio else np.empty((0, 2), dtype=np.int16)
        return self.frame, samples

    def read_ram(self, memory_id=MEMORY_SYSTEM_RAM):
        """Return a snapshot of the core's memory region (default: 2KB NES System RAM)."""
        if not self._loaded:
            return b''
        size = self.lib.retro_get_memory_size(memory_id)
        ptr = self.lib.retro_get_memory_data(memory_id)
        if not ptr or size == 0:
            return b''
        return C.string_at(ptr, size)

    def save_state(self) -> bytes:
        """Serialize current emulator state into raw bytes."""
        if not self._loaded:
            raise RuntimeError('Game not loaded')
        size = self.lib.retro_serialize_size()
        if size == 0:
            raise RuntimeError('Core does not support save states')
        buf = (C.c_ubyte * size)()
        if not self.lib.retro_serialize(buf, size):
            raise RuntimeError('retro_serialize failed')
        return bytes(buf)

    def load_state(self, data: bytes) -> None:
        """Restore emulator state from serialized bytes."""
        if not self._loaded:
            raise RuntimeError('Game not loaded')
        if not data:
            raise ValueError('Empty save state data')
        buf = C.create_string_buffer(data, len(data))
        if not self.lib.retro_unserialize(buf, len(data)):
            raise RuntimeError('retro_unserialize failed')

    def save_state_file(self, path) -> Path:
        """Save serialized emulator state to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.save_state())
        return target

    def load_state_file(self, path) -> None:
        """Load serialized emulator state from disk."""
        self.load_state(Path(path).read_bytes())

    def close(self):
        if self._loaded:
            self.lib.retro_unload_game()
            self._loaded = False
        if self._initialized:
            self.lib.retro_deinit()
            self._initialized = False

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def decode_frame(raw, width, height, pitch, pixel_format):
    """Convert little-endian Libretro packed pixels into RGB uint8 array."""
    bpp = 4 if pixel_format == PIXEL_XRGB8888 else 2
    rows = np.ndarray((height, width), dtype='<u4' if bpp == 4 else '<u2', buffer=raw, strides=(pitch, bpp))
    if pixel_format == PIXEL_XRGB8888:
        r, g, b = (rows >> 16) & 255, (rows >> 8) & 255, rows & 255
    elif pixel_format == PIXEL_RGB565:
        r5, g6, b5 = (rows >> 11) & 31, (rows >> 5) & 63, rows & 31
        r, g, b = (r5 << 3) | (r5 >> 2), (g6 << 2) | (g6 >> 4), (b5 << 3) | (b5 >> 2)
    elif pixel_format == PIXEL_0RGB1555:
        r5, g5, b5 = (rows >> 10) & 31, (rows >> 5) & 31, rows & 31
        r, g, b = (r5 << 3) | (r5 >> 2), (g5 << 3) | (g5 >> 2), (b5 << 3) | (b5 >> 2)
    else:
        raise ValueError('Unsupported pixel format')
    return np.stack((r, g, b), axis=-1).astype(np.uint8)
