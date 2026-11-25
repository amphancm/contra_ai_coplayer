"""SDL2-backed pygame video/input + SDL2 queued PCM audio + V4 State/Dashboard."""
import argparse
import ctypes as C
import ctypes.util
import time
import numpy as np
import pygame
from .dashboard import TerminalDashboard
from .libretro import LibretroHost
from .state import GameState, GameStateDetector

# Libretro joypad IDs
B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R = range(12)


class SDLQueuedAudio:
    def __init__(self, rate):
        libname = ctypes.util.find_library('SDL2')
        if not libname:
            raise RuntimeError('SDL2 library not found (install libsdl2-2.0-0)')
        self.sdl = C.CDLL(libname)

        class Spec(C.Structure):
            _fields_ = [
                ('freq', C.c_int), ('format', C.c_uint16), ('channels', C.c_uint8),
                ('silence', C.c_uint8), ('samples', C.c_uint16), ('padding', C.c_uint16),
                ('size', C.c_uint32), ('callback', C.c_void_p), ('userdata', C.c_void_p),
            ]

        self.sdl.SDL_InitSubSystem.argtypes = [C.c_uint32]
        self.sdl.SDL_InitSubSystem.restype = C.c_int
        self.sdl.SDL_OpenAudioDevice.argtypes = [C.c_char_p, C.c_int, C.POINTER(Spec), C.POINTER(Spec), C.c_int]
        self.sdl.SDL_OpenAudioDevice.restype = C.c_uint32
        self.sdl.SDL_PauseAudioDevice.argtypes = [C.c_uint32, C.c_int]
        self.sdl.SDL_QueueAudio.argtypes = [C.c_uint32, C.c_void_p, C.c_uint32]
        self.sdl.SDL_QueueAudio.restype = C.c_int
        self.sdl.SDL_GetQueuedAudioSize.argtypes = [C.c_uint32]
        self.sdl.SDL_GetQueuedAudioSize.restype = C.c_uint32
        self.sdl.SDL_ClearQueuedAudio.argtypes = [C.c_uint32]
        self.sdl.SDL_CloseAudioDevice.argtypes = [C.c_uint32]
        self.sdl.SDL_QuitSubSystem.argtypes = [C.c_uint32]
        if hasattr(self.sdl, 'SDL_WasInit'):
            self.sdl.SDL_WasInit.argtypes = [C.c_uint32]
            self.sdl.SDL_WasInit.restype = C.c_uint32
            self.owns_audio = not bool(self.sdl.SDL_WasInit(0x10))
        else:
            self.owns_audio = True
        if self.sdl.SDL_InitSubSystem(0x10) != 0:
            raise RuntimeError('SDL audio init failed')
        desired = Spec(int(round(rate)), 0x8010, 2, 0, 1024, 0, 0, None, None)  # AUDIO_S16SYS on little-endian
        obtained = Spec()
        self.device = self.sdl.SDL_OpenAudioDevice(None, 0, C.byref(desired), C.byref(obtained), 0)
        if not self.device:
            self.sdl.SDL_QuitSubSystem(0x10)
            raise RuntimeError('SDL audio device unavailable; try --no-audio')
        self.rate = obtained.freq
        self.sdl.SDL_PauseAudioDevice(self.device, 0)

    def write(self, samples):
        if samples.size == 0:
            return
        # Avoid multi-second latency if render falls behind.
        if self.sdl.SDL_GetQueuedAudioSize(self.device) > self.rate * 4 // 4:
            self.sdl.SDL_ClearQueuedAudio(self.device)
        pcm = np.ascontiguousarray(samples, dtype=np.int16)
        if self.sdl.SDL_QueueAudio(self.device, pcm.ctypes.data, pcm.nbytes) != 0:
            raise RuntimeError('SDL_QueueAudio failed')

    def close(self):
        self.sdl.SDL_CloseAudioDevice(self.device)
        self.sdl.SDL_QuitSubSystem(0x10)


class RuleAI:
    """Baseline P2 policy: shoots, advances, and optionally uses V4 GameState for pit/bullet jumps."""

    def __init__(self, jump_period=85, jump_length=13):
        self.t = 0
        self.jump_period = jump_period
        self.jump_length = jump_length

    def act(self, frame, state: GameState | None = None):
        self.t += 1
        keys = {B, RIGHT}
        if self.t % self.jump_period < self.jump_length:
            keys.add(A)
        if state is not None and state.p2.active:
            # Jump when a pit or wall is detected ahead of P2
            if state.obstacles_p2.pit_ahead or state.obstacles_p2.wall_ahead:
                keys.add(A)
            # Dodge close incoming enemy bullets
            for eb in state.enemy_bullets:
                dx = eb.x - state.p2.x
                dy = abs(eb.y - state.p2.y)
                if -12 <= dx <= 36 and dy <= 18:
                    keys.add(A)
                    break
            # Stay closer to P1 if P2 runs far ahead
            if state.p1.active and state.p1_p2_dx > 70:
                keys.discard(RIGHT)
        return keys


KEYBOARD_P1 = {
    pygame.K_z: B, pygame.K_x: A, pygame.K_j: B, pygame.K_k: A, pygame.K_SPACE: A,
    pygame.K_RETURN: START, pygame.K_RSHIFT: SELECT,
    pygame.K_UP: UP, pygame.K_DOWN: DOWN, pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT,
}

# Physical USB HID scancodes: layout-independent (works with Thai & English keyboard layouts)
SCANCODE_P1 = {
    pygame.KSCAN_Z: B,
    pygame.KSCAN_X: A,
    pygame.KSCAN_J: B,
    pygame.KSCAN_K: A,
    pygame.KSCAN_SPACE: A,
    pygame.KSCAN_RETURN: START,
    pygame.KSCAN_RSHIFT: SELECT,
    pygame.KSCAN_UP: UP,
    pygame.KSCAN_DOWN: DOWN,
    pygame.KSCAN_LEFT: LEFT,
    pygame.KSCAN_RIGHT: RIGHT,
}

# Most SDL gamepads expose button 0 as south face, button 1 as east face.
GAMEPAD_BUTTONS = {0: B, 1: A, 6: SELECT, 7: START}


def human_input(joystick, held_scancodes: set[int] | None = None):
    pressed = pygame.key.get_pressed()
    result = {v for k, v in KEYBOARD_P1.items() if pressed[k]}
    if held_scancodes:
        for sc in held_scancodes:
            btn = SCANCODE_P1.get(sc)
            if btn is not None:
                result.add(btn)
    if joystick is None:
        return result
    for index, button in GAMEPAD_BUTTONS.items():
        if index < joystick.get_numbuttons() and joystick.get_button(index):
            result.add(button)
    if joystick.get_numaxes() >= 2:
        x, y = joystick.get_axis(0), joystick.get_axis(1)
        if x < -0.45: result.add(LEFT)
        if x > 0.45: result.add(RIGHT)
        if y < -0.45: result.add(UP)
        if y > 0.45: result.add(DOWN)
    if joystick.get_numhats():
        x, y = joystick.get_hat(0)
        if x < 0: result.add(LEFT)
        if x > 0: result.add(RIGHT)
        if y < 0: result.add(DOWN)
        if y > 0: result.add(UP)
    return result


def draw_debug_overlay(screen: pygame.Surface, state: GameState, frame_shape: tuple[int, int, int]) -> None:
    """Draw P1/P2/Enemy/Bullet boxes and P1-P2 distance vector onto the pygame display."""
    if state.phase != "PLAYING":
        return
    fh, fw = frame_shape[0], frame_shape[1]
    sw, sh = screen.get_size()
    sx, sy = sw / fw, sh / fh

    def to_screen(x: int, y: int) -> tuple[int, int]:
        return int(x * sx), int(y * sy)

    if state.p1.active and state.p2.active:
        pygame.draw.line(screen, (255, 255, 0), to_screen(state.p1.x, state.p1.y), to_screen(state.p2.x, state.p2.y), 2)
    if state.p1.active:
        px, py = to_screen(state.p1.x, state.p1.y)
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(px - 10, py - 18, 20, 36), 2)
    if state.p2.active:
        px, py = to_screen(state.p2.x, state.p2.y)
        pygame.draw.rect(screen, (0, 220, 255), pygame.Rect(px - 10, py - 18, 20, 36), 2)
    for e in state.enemies:
        ex, ey = to_screen(e.x, e.y)
        pygame.draw.rect(screen, (255, 60, 60), pygame.Rect(ex - 9, ey - 14, 18, 28), 2)
    for eb in state.enemy_bullets:
        bx, by = to_screen(eb.x, eb.y)
        pygame.draw.circle(screen, (255, 165, 0), (bx, by), 5, 2)
    for pb in state.player_bullets:
        bx, by = to_screen(pb.x, pb.y)
        pygame.draw.circle(screen, (255, 255, 255), (bx, by), 3, 1)


def run(args):
    pygame.init()
    pygame.joystick.init()
    joystick = None
    input_device = 'Keyboard'
    if pygame.joystick.get_count():
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        input_device = joystick.get_name()
    dashboard = TerminalDashboard(refresh_hz=args.dashboard_hz, enabled=args.dashboard)
    if not (dashboard.enabled and dashboard.is_tty):
        if joystick:
            print('P1 gamepad:', input_device)
        else:
            print('P1: keyboard (arrows, Z=B/shoot, X=A/jump, Enter=start, Right Shift=select)')
    audio = None
    try:
        with LibretroHost(args.core, args.rom, args.system_dir, args.save_dir) as core:
            fps = core.av.timing.fps
            if not (dashboard.enabled and dashboard.is_tty):
                print(f'Core timing: {fps:.4f} FPS, {core.av.timing.sample_rate:.0f} Hz')
            if not args.no_audio:
                audio = SDLQueuedAudio(core.av.timing.sample_rate)
            screen = pygame.display.set_mode((256 * args.scale, 240 * args.scale), pygame.RESIZABLE)
            pygame.display.set_caption('Contra AI Co-Player V5 | P1 human / P2 AI')
            if args.ai and args.model:
                from .policy import PPOPolicy
                ai = PPOPolicy(args.model)
                ai_label = 'PPO'
            elif args.ai:
                ai = RuleAI()
                ai_label = 'Rule'
            else:
                ai = None
                ai_label = 'OFF'
            detector = GameStateDetector()
            last_frame = None
            last_state = None
            next_tick = time.perf_counter()
            fps_window_start = next_tick
            fps_window_frames = 0
            actual_fps = fps
            frame_period = 1.0 / fps
            frame_count = 0
            held_scancodes: set[int] = set()
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        held_scancodes.add(event.scancode)
                    elif event.type == pygame.KEYUP:
                        held_scancodes.discard(event.scancode)
                    elif event.type == pygame.WINDOWFOCUSLOST:
                        held_scancodes.clear()
                    if event.type == pygame.JOYDEVICEADDED and joystick is None:
                        joystick = pygame.joystick.Joystick(event.device_index)
                        joystick.init()
                        input_device = joystick.get_name()
                    if event.type == pygame.JOYDEVICEREMOVED and joystick and event.instance_id == joystick.get_instance_id():
                        joystick = None
                        input_device = 'Keyboard'
                if not running:
                    break
                p1 = human_input(joystick, held_scancodes)
                p2 = ai.act(last_frame, last_state) if ai is not None else set()
                if args.auto_start and frame_count < 2:
                    p1.add(START)
                    p2.add(START)
                if args.auto_2p:
                    if frame_count in (60, 61):
                        p1.add(START)
                    elif frame_count in (80, 81):
                        p1.add(SELECT)
                    elif frame_count in (100, 101):
                        p1.add(START)
                frame, samples = core.step(p1, p2)
                ram = core.read_ram()
                state = detector.detect(frame, ram, frame_idx=frame_count)
                last_state = state
                if audio:
                    audio.write(samples)
                if frame is not None:
                    last_frame = frame
                    surface = pygame.image.frombuffer(
                        np.ascontiguousarray(frame).tobytes(), (frame.shape[1], frame.shape[0]), 'RGB'
                    )
                    scaled = pygame.transform.scale(surface, screen.get_size())
                    screen.blit(scaled, (0, 0))
                    if args.debug_overlay:
                        draw_debug_overlay(screen, state, frame.shape)
                    pygame.display.flip()
                frame_count += 1
                fps_window_frames += 1
                now = time.perf_counter()
                elapsed = now - fps_window_start
                if elapsed >= 0.5:
                    actual_fps = fps_window_frames / elapsed
                    fps_window_start = now
                    fps_window_frames = 0
                dashboard.update(
                    state,
                    actual_fps=actual_fps,
                    target_fps=fps,
                    p1_keys=p1,
                    p2_keys=p2,
                    audio_on=audio is not None,
                    ai_on=args.ai,
                    ai_label=ai_label,
                    overlay_on=args.debug_overlay,
                    input_device=input_device,
                )
                if args.frames and frame_count >= args.frames:
                    break
                next_tick += frame_period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.25:
                    next_tick = time.perf_counter()
    finally:
        dashboard.close()
        if audio:
            audio.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Contra AI Co-Player V4 Libretro frontend')
    parser.add_argument('--core', required=True, help='Path to fceumm_libretro.so')
    parser.add_argument('--rom', required=True, help='Path to legally obtained Contra NES ROM')
    parser.add_argument('--system-dir')
    parser.add_argument('--save-dir')
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--frames', type=int, default=0, help='Stop after N frames (0=infinite)')
    parser.add_argument('--no-audio', action='store_true')
    parser.add_argument('--no-ai', dest='ai', action='store_false')
    parser.add_argument('--model', help='Path to trained PPO .zip model for P2 AI')
    parser.add_argument('--auto-start', action='store_true', help='Press Start for both ports on first two frames')
    parser.add_argument('--auto-2p', action='store_true', help='Auto-select 2-Player mode on title screen')
    parser.add_argument('--no-dashboard', dest='dashboard', action='store_false', help='Disable terminal dashboard')
    parser.add_argument('--dashboard-hz', type=float, default=10.0, help='Terminal dashboard refresh rate (Hz)')
    parser.add_argument('--debug-overlay', action='store_true', help='Draw V4 detection boxes on game window')
    parser.set_defaults(ai=True, dashboard=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
