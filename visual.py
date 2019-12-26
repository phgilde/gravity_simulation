import pygame


def text_objects(text, font):
    textSurface = font.render(text, True, (255, 255, 255, 100))
    return textSurface, textSurface.get_rect()


def button(display, msg, x, y, w, h, ic, ac):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        pygame.draw.rect(display, ac, (x, y, w, h))

        if click[0] == 1:
            return True
    else:
        pygame.draw.rect(display, ic, (x, y, w, h))

    smallText = pygame.font.SysFont("Arial", 20)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ((x + (w / 2)), (y + (h / 2)))
    display.blit(textSurf, textRect)

    return False
