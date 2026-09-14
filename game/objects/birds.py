import numpy as np
import game.configs as configs
from .bird import Bird
from game.layer import Layer
from game.objects.column import Column
from game.objects.floor import Floor


class Birds:
    def __init__(self, n, sprites_group,render=False):
        self.n = n
        self.render = render
        self.y = np.full(n, configs.SCREEN_HEIGHT//2, dtype=np.float32)
        self.velocity = np.zeros(n, dtype=np.float32)
        self.alive = np.ones(n, dtype=bool)
        self.sprites_group = sprites_group
        self.sprites = [Bird(i, sprites_group) for i in range(n)]

    def reset(self):
        self.y[:] = configs.SCREEN_HEIGHT // 2
        self.velocity[:] = 0
        self.alive[:] = True
        for sprite in self.sprites:
            sprite.rect.y = configs.SCREEN_HEIGHT // 2
            sprite.rect.x = 50 

    def flap(self, actions):
        flap_mask = (actions == 1) & self.alive
        self.velocity[flap_mask] = -6

    def physics_step(self):
        self.velocity[self.alive] += configs.GRAVITY
        self.y[self.alive] += self.velocity[self.alive]

    def sync_to_sprites(self):
        # push vectorized state into pygame objects, only for rendering/collision
        for i, sprite in enumerate(self.sprites):
            sprite.rect.y = int(self.y[i])
            sprite.flap = self.velocity[i]

    def check_collisions(self, pipe_rects):
        # pipe_rects: array of shape (num_pipes, 4) -> [x, y, w, h]
        bird_x = np.array([s.rect.x for s in self.sprites])
        bird_y = self.y
        bird_w = self.sprites[0].rect.width
        bird_h = self.sprites[0].rect.height

        out_of_bounds = (bird_y > 400) | (bird_y < 0)
        self.alive[out_of_bounds] = False

        #CHECK THIS LATER
        if self.render:  # Safely check render flag
            for i in np.flatnonzero(out_of_bounds):
                self.sprites[i].rect.x = -50
                self.sprites[i].rect.y = 50


        # broadcast: (population, 1) vs (1, num_pipes) -> (population, num_pipes) overlap matrix
        x_overlap = (bird_x[:, None] < pipe_rects[:, 0] + pipe_rects[:, 2]) & \
                    (bird_x[:, None] + bird_w > pipe_rects[:, 0])
        y_overlap = (bird_y[:, None] < pipe_rects[:, 1] + pipe_rects[:, 3]) & \
                    (bird_y[:, None] + bird_h > pipe_rects[:, 1])

        broad_phase_hit = (x_overlap & y_overlap).any(axis=1)  # shape (population,)

        # Only run pygame's expensive per-pixel mask check on birds that plausibly collided
        for i in np.flatnonzero(broad_phase_hit & self.alive):
            if self.sprites[i].check_collision(self.sprites_group):
                self.alive[i] = False
            if not self.alive[i] and self.render:
                self.sprites[i].rect.x = -50
                self.sprites[i].rect.y = 50
