import os
import pygame
import numpy as np

import assets 
import configs 
from objects.background import Background
from objects.birds import Birds
from objects.column import Column
from objects.floor import Floor


class FlappyBirdEnv:
    COLUMN_SPAWN_EVERY_N_STEPS = 100

    def __init__(self, population_size=10, render=True):
        self.render_enabled = render
        self.population_size = population_size

        if not self.render_enabled:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            
        pygame.init()
        assets.load_sprites() 

        if self.render_enabled:
            self.screen = pygame.display.set_mode((configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird Engine v1.0")
            assets.load_audios()
            self.sprites = pygame.sprite.LayeredUpdates()
            Background(0, self.sprites)
            Background(1, self.sprites)
            Floor(0, self.sprites)
            Floor(1, self.sprites)
        else:
            self.screen = pygame.display.set_mode((1, 1)) 
            self.sprites = pygame.sprite.Group()


        self.clock = pygame.time.Clock()

        self.birds = Birds(population_size, self.sprites, self.render_enabled)

        self.scores = np.zeros(population_size, dtype=np.int32)
        self.kill_count = 0
        self.step_count = 0

        self.reset()

    def reset(self):
        # Clear only columns between episodes; background/floor/birds persist.
        for sprite in list(self.sprites):
            if isinstance(sprite, Column):
                sprite.kill()

        self.birds.reset()
        self.scores = np.zeros(self.population_size, dtype=np.int32)
        self.kill_count = 0
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        bird_states = np.zeros((self.population_size, 2), dtype=np.float32)
        alive = self.birds.alive
        bird_states[alive, 0] = self.birds.y[alive]
        bird_states[alive, 1] = self.birds.velocity[alive]

        pipe_states = sorted(
            [[s.rect.x, s.rect.y] for s in self.sprites if isinstance(s, Column)],
            key=lambda p: p[0]
        )
        upcoming = [p for p in pipe_states if p[0] > 25]

        # Need the two nearest upcoming pipes; pad with a fallback if fewer exist.
        if len(upcoming) >= 2:
            pipe1, pipe2 = upcoming[0], upcoming[1]
        elif len(upcoming) == 1:
            pipe1 = upcoming[0]
            pipe2 = upcoming[0]   # duplicate nearest pipe as a placeholder for the (not-yet-spawned) next one
        elif pipe_states:
            pipe1 = pipe2 = pipe_states[0]
        else:
            pipe1 = pipe2 = [0, 0]

        broadcasted_pipes = np.tile([pipe1[0], pipe1[1], pipe2[0], pipe2[1]], (self.population_size, 1))
        return np.hstack((bird_states, broadcasted_pipes)).astype(np.float32)

    def step(self, actions):
        """
        actions: array-like of length population_size, 0 = no-op, 1 = flap
        returns: (state_matrix, alive_mask, scores, done)
        """
        actions = np.asarray(actions)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit("Window closed")

        if self.step_count % self.COLUMN_SPAWN_EVERY_N_STEPS == 0:
            Column(self.sprites)

        # Update non-bird, non-column sprites (background scroll, floor scroll)
        for sprite in self.sprites:
            if isinstance(sprite, Column) or sprite in self.birds.sprites:
                continue
            sprite.update()

        # Update columns and collect their rects for vectorized collision checks
        pipe_rects = []
        for sprite in self.sprites:
            if isinstance(sprite, Column):
                sprite.update()
                pipe_rects.append([sprite.rect.x, sprite.rect.y, sprite.rect.width, sprite.rect.height])
        pipe_rects = np.array(pipe_rects, dtype=np.float32) if pipe_rects else np.empty((0, 4), dtype=np.float32)

        prev_alive_count = int(self.birds.alive.sum())

        # Bird step: vectorized flap + physics, then collision check against columns
        self.birds.flap(actions)
        self.birds.physics_step()
        self.birds.check_collisions(pipe_rects)
        self.birds.sync_to_sprites()

        newly_dead = prev_alive_count - int(self.birds.alive.sum())
        if newly_dead > 0 and self.render_enabled:
            assets.play_audio("hit")
        self.kill_count += newly_dead

        # Scoring — vectorized increment for all currently-alive birds
        for sprite in self.sprites:
            if type(sprite) is Column and sprite.is_passed():
                self.scores[self.birds.alive] += 1
                if self.render_enabled:
                    assets.play_audio("point")

        if self.render_enabled:
            self.screen.fill(0)
            self.sprites.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(configs.FPS)

        self.step_count += 1

        state = self._get_state()
        done = bool(self.kill_count >= self.population_size)

        return state, self.birds.alive.copy(), self.scores.copy(), done

    def close(self):
        pygame.quit()


# if __name__ == "__main__":
#     import time

#     env = FlappyBirdEnv(population_size=20, render=True)
#     state = env.reset()
#     print("Initial state:\n", state)
#     start = time.time()
#     done = False
#     while not done :
#         actions = np.zeros(env.population_size, dtype=int)  # no-op
#         if time.time() - start > 0.5:
#             actions = np.random.randint(0, 2, env.population_size)  # random flaps after 1 second
#             start = time.time()
#         state, alive, scores, done = env.step(actions)
#         print("state:\n", state, "\nalive: ",alive , "scores:", scores)
#         time.sleep(0.02)  # simulate manual/slow calling; safe because spawning is step-based now

#     print("All birds dead, resetting.")
#     state = env.reset()

#     env.close()