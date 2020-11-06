import pygame
import pygame.freetype
import torch
import numpy as np

from lib.displayer_mnistmixer import displayer
from lib.utils import mat_distance, make_seed
from lib.GNCAModel import model_VAE

map_size = (28,28)
pix_size = 8
pen_radius = 1
max_pen_radius = 2

ALPHA_CHANNEL = 0
HIDDEN = 32
CHANNEL_N = 8
HIDDEN_CHANNEL_N = 64
N_STEPS = 64

DEVICE = torch.device("cpu")
model_path = "models/gen_AE_mnist.pth"
init_coord = (map_size[0]//2, map_size[1]//2)
my_model = model_VAE(HIDDEN, CHANNEL_N, ALPHA_CHANNEL, HIDDEN_CHANNEL_N, device=DEVICE)
my_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def redo_layout(disp, map_size, pix_size, pen_radius, myfont, myfont_s):
    src0 = pygame.draw.rect(disp.screen, (255,255,255), (0,0,
                                                  map_size[1]*pix_size,map_size[0]*pix_size), 1)
    src1 = pygame.draw.rect(disp.screen, (255,255,255), (map_size[1]*pix_size+2*pix_size,0,
                                                  map_size[1]*pix_size,map_size[0]*pix_size), 1)
    pygame.draw.rect(disp.screen, (255,255,255), (map_size[1]*pix_size*2+4*pix_size,0,
                                                  map_size[1]*pix_size,map_size[0]*pix_size), 1)

    textsurface, _ = myfont.render('Source 1', (255,255,255))
    text_width, text_height = textsurface.get_width(), textsurface.get_height()
    disp.screen.blit(textsurface,(map_size[1]*pix_size*0.5-text_width/2,
                                  map_size[0]*pix_size+2*pix_size-text_height/2))
    textsurface, _ = myfont.render('Source 2', (255,255,255))
    text_width, text_height = textsurface.get_width(), textsurface.get_height()
    disp.screen.blit(textsurface,(map_size[1]*pix_size*1.5+2*pix_size-text_width/2,
                                  map_size[0]*pix_size+2*pix_size-text_height/2))
    textsurface, _ = myfont.render('Output', (255,255,255))
    text_width, text_height = textsurface.get_width(), textsurface.get_height()
    disp.screen.blit(textsurface,(map_size[1]*pix_size*2.5+4*pix_size-text_width/2,
                                  map_size[0]*pix_size+2*pix_size-text_height/2))

    btn_width, btn_height = map_size[1]*0.75*pix_size, 4*pix_size
    btn_clc_0 = pygame.draw.rect(disp.screen, (255,255,255),
                                 (map_size[1]*pix_size*0.5-btn_width/2,
                                  map_size[0]*pix_size+6*pix_size-btn_height/2,
                                  btn_width, btn_height), 2)
    textsurface, _ = myfont.render('Clear', (255,255,255))
    text_width, text_height = textsurface.get_width(), textsurface.get_height()
    disp.screen.blit(textsurface,(map_size[1]*pix_size*0.5-text_width/2,
                                  map_size[0]*pix_size+6*pix_size-text_height/2))
    btn_clc_1 = pygame.draw.rect(disp.screen, (255,255,255),
                                 (map_size[1]*pix_size*1.5+2*pix_size-btn_width/2,
                                  map_size[0]*pix_size+6*pix_size-btn_height/2,
                                  btn_width, btn_height), 2)
    textsurface, _ = myfont.render('Clear', (255,255,255))
    text_width, text_height = textsurface.get_width(), textsurface.get_height()
    disp.screen.blit(textsurface,(map_size[1]*pix_size*1.5+2*pix_size-text_width/2,
                                  map_size[0]*pix_size+6*pix_size-text_height/2))
    btn_gen = pygame.draw.rect(disp.screen, (255,255,255),
                               (map_size[1]*pix_size*2.5+4*pix_size-btn_width/2,
                                map_size[0]*pix_size+6*pix_size-btn_height/2,
                                btn_width, btn_height), 2)
    textsurface, _ = myfont.render('Generate', (255,255,255))
    text_width, text_height = textsurface.get_width(), textsurface.get_height()
    disp.screen.blit(textsurface,(map_size[1]*pix_size*2.5+4*pix_size-text_width/2,
                                  map_size[0]*pix_size+6*pix_size-text_height/2))

    btn_pen = pygame.draw.rect(disp.screen, (255,255,255),
                               (map_size[1]*pix_size*0.5-btn_width/2,
                                map_size[0]*pix_size+12*pix_size-btn_height/2,
                                btn_width, btn_height), 2)
    textsurface, _ = myfont.render('Brush Size '+str(pen_radius), (255,255,255))
    text_width, text_height = textsurface.get_width(), textsurface.get_height()
    disp.screen.blit(textsurface,(map_size[1]*pix_size*0.5-text_width/2,
                                  map_size[0]*pix_size+12*pix_size-text_height/2))

    textsurface, _ = myfont_s.render('Leave a canvas blank if you only wish to mimic one of your masterpiece.', (255,255,255))
    text_width, text_height = textsurface.get_width(), textsurface.get_height()
    disp.screen.blit(textsurface,(map_size[1]*pix_size+2*pix_size,
                                  map_size[0]*pix_size+12*pix_size-1.2*text_height))

    textsurface, _ = myfont_s.render('Draw both if you want to mix them.', (255,255,255))
    text_width, text_height = textsurface.get_width(), textsurface.get_height()
    disp.screen.blit(textsurface,(map_size[1]*pix_size+2*pix_size,
                                  map_size[0]*pix_size+12*pix_size+0.8*text_height))

    # pygame.draw.line(disp.screen, (255,255,255),
    #                  (0, map_size[0]*pix_size+16*pix_size),
    #                  (map_size[1]*pix_size*3+4*pix_size, map_size[0]*pix_size+16*pix_size))
    pygame.display.update()
    return src0, src1, btn_clc_0, btn_clc_1, btn_gen, btn_pen

_rows = np.arange(map_size[0]).repeat(map_size[1]).reshape(map_size)
_cols = np.arange(map_size[1]).reshape([1,-1]).repeat(map_size[0],axis=0)
_map_pos = np.array([_rows,_cols]).transpose([1,2,0])
_map0 = np.zeros(map_size)
_map1 = np.zeros(map_size)
_mapr = np.zeros(map_size)

disp = displayer(map_size, pix_size)
myfont = pygame.freetype.SysFont('Comic Sans MS', 20)
myfont_s = pygame.freetype.SysFont('Comic Sans MS', 12)
disp.draw_all(_map0, 0)
disp.draw_all(_map1, 0)
srcs = redo_layout(disp, map_size, pix_size, pen_radius, myfont, myfont_s)
src0, src1, btn_clc_0, btn_clc_1, btn_gen, btn_pen = srcs

isMouseDown = False
running = True
history = []
while running:

    if len(history)>0:
        if len(history)==1:
            disp.screen.fill((0,0,0))
            disp.draw_all(_map0, 0)
            disp.draw_all(_map1, 1)
        disp.draw_all(history.pop()[0,...,0], 2)
        srcs = redo_layout(disp, map_size, pix_size, pen_radius, myfont, myfont_s)
        src0, src1, btn_clc_0, btn_clc_1, btn_gen, btn_pen = srcs
        pygame.event.pump()
        continue

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                isMouseDown = True
            if btn_clc_0.collidepoint(event.pos):
                _map0 = np.zeros(map_size)
                disp.draw_all(_map0, 0)
                srcs = redo_layout(disp, map_size, pix_size, pen_radius, myfont, myfont_s)
                src0, src1, btn_clc_0, btn_clc_1, btn_gen, btn_pen = srcs
            if btn_clc_1.collidepoint(event.pos):
                _map1 = np.zeros(map_size)
                disp.draw_all(_map1, 1)
                srcs = redo_layout(disp, map_size, pix_size, pen_radius, myfont, myfont_s)
                src0, src1, btn_clc_0, btn_clc_1, btn_gen, btn_pen = srcs
            if btn_pen.collidepoint(event.pos):
                pen_radius+=1
                if pen_radius>max_pen_radius:
                    pen_radius=0
                disp.screen.fill((0,0,0))
                disp.draw_all(_map0, 0)
                disp.draw_all(_map1, 1)
                disp.draw_all(_mapr, 2)
                srcs = redo_layout(disp, map_size, pix_size, pen_radius, myfont, myfont_s)
                src0, src1, btn_clc_0, btn_clc_1, btn_gen, btn_pen = srcs
            if btn_gen.collidepoint(event.pos):
                if np.sum(_map0)>0 or np.sum(_map1)>0:
                    seed = make_seed(map_size, CHANNEL_N, np.arange(CHANNEL_N-ALPHA_CHANNEL)+ALPHA_CHANNEL, init_coord)
                    x0 = np.repeat(seed[None, ...], 1, 0)
                    x0 = torch.from_numpy(x0.astype(np.float32)).to(DEVICE)
                    if np.sum(_map0)>0 and np.sum(_map1)==0:
                        print("mimicing map0")
                        xs = [torch.from_numpy(_map0.reshape([1,-1]).astype(np.float32)).to(DEVICE),
                              torch.from_numpy(_map0.reshape([1,-1]).astype(np.float32)).to(DEVICE)]
                    elif np.sum(_map0)==0 and np.sum(_map1)>0:
                        print("mimicing map1")
                        xs = [torch.from_numpy(_map1.reshape([1,-1]).astype(np.float32)).to(DEVICE),
                              torch.from_numpy(_map1.reshape([1,-1]).astype(np.float32)).to(DEVICE)]
                    else:
                        print("mimicing both")
                        xs = [torch.from_numpy(_map0.reshape([1,-1]).astype(np.float32)).to(DEVICE),
                              torch.from_numpy(_map1.reshape([1,-1]).astype(np.float32)).to(DEVICE)]
                    y, history, zs = my_model.multi_infer(x0, xs, N_STEPS)
                    _mapr = y.detach().cpu().numpy()[0,...,0]
                    print(zs[0][0])
                    print(zs[1][0])
                    history.reverse()

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                isMouseDown = False

    if isMouseDown:
        try:
            if src0.collidepoint(event.pos):
                mouse_pos = np.array([int(event.pos[1]/pix_size),
                                      int(event.pos[0]/pix_size)])
                draw = (mat_distance(_map_pos, mouse_pos)<=pen_radius).reshape([map_size[0],map_size[1],1]).astype(float)
                changed = np.argwhere(draw>0)[...,:2]
                for i,j in changed: _map0[i,j] = 1
                disp.draw(_map0, changed, 0)
            if src1.collidepoint(event.pos):
                mouse_pos = np.array([int(event.pos[1]/pix_size),
                                      int((event.pos[0]-(map_size[0]*pix_size+2*pix_size))/pix_size)])
                draw = (mat_distance(_map_pos, mouse_pos)<=pen_radius).reshape([map_size[0],map_size[1],1]).astype(float)
                changed = np.argwhere(draw>0)[...,:2]
                for i,j in changed: _map1[i,j] = 1
                disp.draw(_map1, changed, 1)
        except AttributeError:
            pass
