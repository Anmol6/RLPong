import pygame, sys, math
from pygame.locals import *
from random import randint

pygame.init()
#size = width, height = 320, 240
#speed = [2, 2]
#black = 0, 0, 0
#screen = pygame.display.set_mode(size)
#ball = pygame.image.load("ball.gif")
#ballrect = ball.get_rect()
#Create objects
#Set screen size
size = width,height = 600,400
screen = pygames.display.set_mode(size)
#Set Background
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill((200,150,120))

speedb1 = [0,1]
speedb2 = [0,1]
speedball = [2,0]
bar1x, bar2x = 20.,590.
bar1y, bar2y = 200.,200.
class Ball(pygame.sprite.Sprite):
	''' Ball that will be used in the ping pong game
	Returns: a ball objects
	'''
	def _init_(self,vector):
		pygame.sprite.Sprite._init_(self)
		ball_image = pygame.image.load('ball.gif')
		self.image = ball_image.convert()
		self.rect = self.image.get_rect()
		screen = pygame.display.get_surface()

class Paddle(pygame.sprite)




while 1:
	for event in pygame.event.get():
	    if event.type == pygame.QUIT: sys.exit()

        if event.type == pygame.
        ballrect = ballrect.move(speed)
        if ballrect.left < 0 or ballrect.right > width:
            speed[0] = -speed[0]
        if ballrect.top < 0 or ballrect.bottom > height:
            speed[1] = -speed[1]

        screen.fill(black)
        screen.blit(ball, ballrect)
        pygame.display.flip()

