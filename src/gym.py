import os
import pygame
import numpy as np
import itertools
import sys
import game.assets as assets
import game.configs as configs 
from game.objects.background import Background
from game.objects.birds import Birds
from game.objects.column import Column
from game.objects.floor import Floor


class FlappyBirdEnv:
    COLUMN_SPAWN_EVERY_N_STEPS = 100
    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])

    def __init__(self, population_size, render=True,display_score=False):
        self.render_enabled = render
        self.population_size = population_size
        self.display_score = display_score

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
            Floor(0, self.sprites)
            Floor(1, self.sprites)

        self.clock = pygame.time.Clock()

        self.birds = Birds(population_size, self.sprites, self.render_enabled)
        self.columns = []
        self.scores = np.zeros(population_size, dtype=np.int32)
        self.kill_count = 0
        self.step_count = 0
        self.current_frame = next(self.spinner)
        self.reset()

    def reset(self):
        # Clear only columns between episodes; background/floor/birds persist.
        for sprite in self.columns:
            sprite.kill()
        self.columns.clear()

        self.birds.reset()
        self.scores = np.zeros(self.population_size, dtype=np.int32)
        self.kill_count = 0
        self.step_count = 0
        return np.column_stack((self._get_state(),np.zeros(self.population_size)))

    def _get_state(self):
        bird_states = np.zeros((self.population_size, 2), dtype=np.float32)
        alive = self.birds.alive
        bird_states[alive, 0] = self.birds.y[alive]
        bird_states[alive, 1] = self.birds.velocity[alive]

        pipe_states = sorted([[c.rect.x, c.rect.y] for c in self.columns], key=lambda p: p[0])
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
        return np.hstack((bird_states, broadcasted_pipes)).astype(np.int32)

    def step(self, actions):
        """
        actions: array-like of length population_size, 0 = no-op, 1 = flap
        returns: (state_matrix, alive_mask, scores, done)
        """
        actions = np.asarray(actions)

        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit("Window closed")

        if self.step_count % self.COLUMN_SPAWN_EVERY_N_STEPS == 0:
            new_col = Column(self.sprites)
            self.columns.append(new_col)

        # Update non-bird, non-column sprites (background scroll, floor scroll)
        if self.render_enabled:
            for sprite in self.sprites:
                if isinstance(sprite, Column) or sprite in self.birds.sprites:
                    continue
                sprite.update()

        self.columns = [col for col in self.columns if col.alive()]
        # Update columns and collect their rects for vectorized collision checks
        pipe_rects = []
        for col in self.columns:
            col.update()
            pipe_rects.append([col.rect.x, col.rect.y, col.rect.width, col.rect.height])
            
        pipe_rects = np.array(pipe_rects, dtype=np.float32) if pipe_rects else np.empty((0, 4), dtype=np.float32)

        prev_alive_count = int(self.birds.alive.sum())

        self.birds.flap(actions)
        self.birds.physics_step()
        self.birds.check_collisions(pipe_rects)
        self.birds.sync_to_sprites()

        newly_dead = prev_alive_count - int(self.birds.alive.sum())
        if newly_dead > 0 and self.render_enabled:
            assets.play_audio("hit")
        self.kill_count += newly_dead

        # # Scoring — vectorized increment for all currently-alive birds
        for sprite in self.columns:
            if sprite.is_passed():  
                self.scores[self.birds.alive] += 1
                if self.render_enabled:
                    assets.play_audio("point")


        if self.display_score:
            if self.step_count%500 == 0:
                self.current_frame = next(self.spinner)

            max_score = self.scores.max()
            print(f"\r{self.current_frame} Flying... Max Score: {max_score}\033[K", end="", flush=True)


        if self.render_enabled:
            self.screen.fill(0)
            self.sprites.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(configs.FPS)

        self.step_count += 1

        state = self._get_state()
        state = np.column_stack((state,actions))
        done = bool(self.kill_count >= self.population_size)

        return state, self.birds.alive.copy(), self.scores.copy(), done

    def close(self):
        pygame.quit()
