import os

# Set BEFORE importing pygame if you want headless (no window) training.
# Comment this out when you want to actually watch the birds play.
RENDER = True
if not RENDER:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import numpy as np

import assets
import configs
from objects.background import Background
from objects.bird import Bird
from objects.column import Column
from objects.floor import Floor
from objects.score import Score

pygame.init()

screen = pygame.display.set_mode((configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird Matrix Engine v1.0")

img = pygame.image.load('assets/icons/red_bird.png')
pygame.display.set_icon(img)

clock = pygame.time.Clock()
column_create_event = pygame.USEREVENT
running = True
gamestarted = False

assets.load_sprites()
assets.load_audios()

sprites = pygame.sprite.LayeredUpdates()

POPULATION_SIZE = 10
birds = []
scores = []
kill_count = 0


def create_environment():
    Background(0, sprites)
    Background(1, sprites)
    Floor(0, sprites)
    Floor(1, sprites)


def spawn_population(n):
    global birds, scores, kill_count
    create_environment()
    birds = [Bird(i, sprites) for i in range(n)]
    kill_count = 0
    scores = [0] * n


spawn_population(POPULATION_SIZE)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == column_create_event:
            Column(sprites)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not gamestarted:
                gamestarted = True
                pygame.time.set_timer(column_create_event, 1500)
            if event.key == pygame.K_ESCAPE:
                sprites.empty()
                spawn_population(POPULATION_SIZE)
                gamestarted = False
                pygame.time.set_timer(column_create_event, 0)

    bird_states = np.zeros((POPULATION_SIZE, 2), dtype=np.float32)
    pipe_states = []

    if gamestarted:
        for sprite in sprites:
            if isinstance(sprite, Bird):
                if sprite.alive:
                    b_y, b_vel = sprite.update()
                    bird_states[sprite.index] = [b_y, b_vel]
                # dead birds: skip update entirely, no physics/animation needed
            elif isinstance(sprite, Column):
                pipe_data = sprite.update()
                pipe_states.append(pipe_data)
            else:
                sprite.update()

    active_pipe = [0, 0]
    if pipe_states:
        upcoming_pipes = [p for p in pipe_states if p[0] > 30]
        active_pipe = upcoming_pipes[0] if upcoming_pipes else pipe_states[0]

    broadcasted_pipe = np.tile([active_pipe[0], active_pipe[1]], (POPULATION_SIZE, 1))
    state_matrix = np.hstack((bird_states, broadcasted_pipe))

    actions = np.random.randint(0, 2, POPULATION_SIZE)  # 0 = do nothing, 1 = flap

    if gamestarted:
        for i, bird in enumerate(birds):
            if not bird.alive:
                continue
            if actions[i] == 1:
                bird.flap = -6
                assets.play_audio("wing")

            if bird.check_collision(sprites) or bird.rect.y > configs.SCREEN_HEIGHT or bird.rect.y < 0:
                bird.alive = False
                assets.play_audio("hit")
                kill_count += 1

    # --- 4. Score tracking per bird ---
    for sprite in sprites:
        if type(sprite) is Column and sprite.is_passed():
            for i, bird in enumerate(birds):
                if bird.alive:
                    scores[i] += 1

    # --- 5. Render (skip entirely for fast headless training) ---
    if RENDER:
        screen.fill(0)
        sprites.draw(screen)
        pygame.display.flip()
        clock.tick(configs.FPS)

    if gamestarted and kill_count >= POPULATION_SIZE:
        sprites.empty()
        spawn_population(POPULATION_SIZE)
        gamestarted = False
        pygame.time.set_timer(column_create_event, 0)

pygame.quit()