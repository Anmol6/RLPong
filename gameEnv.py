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
speedb=10
speedb1 = [0,1]
speedb2 = [0,1]
speedball = [1,1]
bar1x, bar2x = 20.,590.
bar1y, bar2y = 200.,200.
class Ball(pygame.sprite.Sprite):
	''' Ball that will be used in the ping pong game
	Returns: a ball objects
	'''
	#initialize sprite object
	def _init_(self,vector):
		pygame.sprite.Sprite._init_(self)
		self.vector = vector
		#load image and convert it
		ball_image = pygame.image.load('ball.gif')
		self.image = ball_image.convert()
		self.rect = self.image.get_rect()
		
		screen = pygame.display.get_surface()
		self.area = screen.get_rect()
	#Update spatial position of ball given velocity vector	
	def updatepos(self):
		angle, z = self.vector
		move_x,move_y = (z*math.cos(angle)), (z*math.sin(angle))
		newpos = self.rect.move(move_x,move_y)


class Paddle(pygame.sprite.Sprite):
	def _init_(self,side,posvector,speedb):
		self.side = side
		pygame.sprite.Sprite._init_(self)
		self.paddle = pygame.Surface(posvector)
		paddle.fill((255,0,0))

		self.rect = self.paddle.get_rect()
		screen = pygame.display.get_surface()
		self.area = screen.get_rect
		self.speed = speedb
		self.state = "still"
		self.re_pos_init()

	def re_pos_init(self):
		self.moveby = [0,0]
		if self.side == "left":
			self.rect.midleft = self.area.midleft

		elif self.side =="right":
			self.rect.midright = self.area.midright
	#this changes the paddle position, based on how moveby has changed 		
	def update(self):
		newpos = self.rect.move(self.moveby)
		if self.area.contains(newpos):
			self.rect = newpos
		pygame.event.pump()


	def moveup(self):
		self.state = "moveup"
		self.moveby[1] = self.moveby[1] + self.speed

	def movedown(self):	
		self.state = "movedown"
		self.moveby[1] = self.moveby[1] - self.speed




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

