import pygame
import numpy as np

class displayer:

    def __init__(self, _map_shape, pix_size, has_gap=False):
        """
        _map_size: tuple
        color_map: a list indicates the color to each index.
                   0 : empty block, should always white
                   1+: varies building types
        """
        pygame.init()
        clock = pygame.time.Clock()
        clock.tick(60)

        self._map_shape = _map_shape
        self.has_gap = has_gap
        self.pix_size = pix_size
        self.screen = pygame.display.set_mode((_map_shape[1]*self.pix_size,
                                               _map_shape[0]*self.pix_size + 100))
        self.color_map = [(255,255,255),
                          (128,255,128),
                          (255,255,128),
                          (255,180, 50),
                          (255,128,128),
                          (255,128,255)]
        self.screen.fill((255,255,255))
        pygame.draw.line(self.screen, (0,0,0),
                         (0, self._map_shape[0]*self.pix_size),
                         (self._map_shape[1]*self.pix_size, self._map_shape[0]*self.pix_size))
        pygame.display.update()

    def clear(self):
        self.screen.fill((255,255,255))
        pygame.draw.line(self.screen, (0,0,0),
                         (0, self._map_shape[0]*self.pix_size),
                         (self._map_shape[1]*self.pix_size, self._map_shape[0]*self.pix_size))
        pygame.display.update()

    def draw_all(self, _map, valid_mask):
        for i in range(_map.shape[0]):
            for j in range(_map.shape[1]):
                x = j * self.pix_size + int(self.pix_size/2)
                y = i * self.pix_size + int(self.pix_size/2)
                if self.has_gap:
                    size = min(int(self.pix_size * 0.75), self.pix_size-2)
                else:
                    size = self.pix_size
                s = pygame.Surface((size,size))
                if valid_mask[i,j,0]:
                    c = self.color_map[(np.argmax(_map[i,j,:5])+1)*(_map[i,j,5]>0.1)]
                else:
                    c = (0,0,0)
                s.fill(c)
                self.screen.blit(s, (x-int(size/2), y-int(size/2)))
        pygame.display.update()

    def draw(self, _map, valid_mask, valid_coords):
        for i,j in valid_coords:
            x = j * self.pix_size + int(self.pix_size/2)
            y = i * self.pix_size + int(self.pix_size/2)
            if self.has_gap:
                size = min(int(self.pix_size * 0.75), self.pix_size-2)
            else:
                size = self.pix_size
            s = pygame.Surface((size,size))
            if valid_mask[i,j,0]:
                c = self.color_map[(np.argmax(_map[i,j,:5])+1)*(_map[i,j,5]>0.1)]
            else:
                c = (0,0,0)
            s.fill(c)
            self.screen.blit(s, (x-int(size/2), y-int(size/2)))
        pygame.display.update()
