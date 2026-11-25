import numpy as np
import pygame
from contra_coplayer.libretro import decode_frame, PIXEL_RGB565, PIXEL_XRGB8888, PIXEL_0RGB1555
from contra_coplayer.app import RuleAI, SCANCODE_P1, B, RIGHT, A

def test_rgb565_with_pitch_padding():
    raw = bytes.fromhex('00f8e0070000')
    result = decode_frame(raw, 2, 1, 6, PIXEL_RGB565)
    assert result.shape == (1, 2, 3)
    assert result[0, 0].tolist() == [255, 0, 0]
    assert result[0, 1].tolist() == [0, 255, 0]

def test_xrgb8888():
    result = decode_frame(bytes.fromhex('ff000000'), 1, 1, 4, PIXEL_XRGB8888)
    assert result[0, 0].tolist() == [0, 0, 255]

def test_1555():
    result = decode_frame(bytes.fromhex('1f00'), 1, 1, 2, PIXEL_0RGB1555)
    assert result[0, 0].tolist() == [0, 0, 255]

def test_rule_ai():
    ai = RuleAI(jump_period=10, jump_length=2)
    assert ai.act(None) == {B, RIGHT, A}
    assert ai.act(None) == {B, RIGHT}


def test_scancode_mapping_layout_independent():
    assert SCANCODE_P1[pygame.KSCAN_Z] == B
    assert SCANCODE_P1[pygame.KSCAN_X] == A
    assert SCANCODE_P1[pygame.KSCAN_SPACE] == A
