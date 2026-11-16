import pygame


def main() -> None:
    pygame.init()

    # Configuración de la ventana
    width = 1000
    height = 700
    fps = 60

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("NeuroEcosistema: Evolución artificial")

    clock = pygame.time.Clock()
    running = True

    # Bucle principal
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fondo del ecosistema
        screen.fill((30, 45, 35))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()