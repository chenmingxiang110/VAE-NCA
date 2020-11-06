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

        self.has_gap = has_gap
        self._map_shape = _map_shape
        self.pix_size = pix_size
        self.screen = pygame.display.set_mode((_map_shape[1]*self.pix_size*3+4*self.pix_size,
                                               _map_shape[0]*self.pix_size+16*self.pix_size))

    def clear(self):
        self.screen.fill((255,255,255))
        pygame.display.update()

    def draw(self, _map, valid_coords, map_index):
        for i,j in valid_coords:
            x_jump = map_index*(self._map_shape[1]*self.pix_size+2*self.pix_size)
            x = j * self.pix_size + int(self.pix_size/2) + x_jump
            y = i * self.pix_size + int(self.pix_size/2)
            if self.has_gap:
                size = min(int(self.pix_size * 0.75), self.pix_size-2)
            else:
                size = self.pix_size
            s = pygame.Surface((size,size))
            c = np.clip((_map[i,j]*np.ones(3)*256).astype(int), 0, 255)
            s.fill(c)
            self.screen.blit(s, (x-int(size/2), y-int(size/2)))
        pygame.display.update()

    def draw_all(self, _map, map_index):
        for i in range(_map.shape[0]):
            for j in range(_map.shape[1]):
                x_jump = map_index*(self._map_shape[1]*self.pix_size+2*self.pix_size)
                x = j * self.pix_size + int(self.pix_size/2) + x_jump
                y = i * self.pix_size + int(self.pix_size/2)
                if self.has_gap:
                    size = min(int(self.pix_size * 0.75), self.pix_size-2)
                else:
                    size = self.pix_size
                s = pygame.Surface((size,size))
                c = np.clip((_map[i,j]*np.ones(3)*256).astype(int), 0, 255)
                s.fill(c)
                self.screen.blit(s, (x-int(size/2), y-int(size/2)))
        pygame.display.update()
