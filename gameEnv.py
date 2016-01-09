import pygame, sys, math
from pygame.locals import *
from random import randint


#size = width, height = 320, 240
#speed = [2, 2]
#black = 0, 0, 0
#screen = pygame.display.set_mode(size)
#ball = pygame.image.load("ball.gif")
#ballrect = ball.get_rect()
#Create objects
#Set screen size

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
		self.hitpaddle = 0
		screen = pygame.display.get_surface()
		self.area = screen.get_rect()
	#Update spatial position of ball given velocity vector	
	def updatepos(self):
		angle, vel = self.vector
		move_x,move_y = (vel*math.cos(angle)), (vel*math.sin(angle))
		newpos = self.rect.move(move_x,move_y)

		#looking at collisions with sides of frame
		if not self.area.contains(newpos):
			tl_out = not self.area.collidepoint(newpos.topleft)
			tr_out = not self.area.collidepoint(newpos.topright)
			bl_out = not self.area.collidepoint(newpos.bottomleft)
			br_out = not self.area.collidepoint(newpos.bottomright)
			#if ball goes from sides
			if (tl_out and tr_out) or (bl_out and br_out):
				angle = -angle
			#if ball goes past any of the players, ball is put on the other side of the court	
			if (tl_out and bl_out):
				self.offcourt(player=2)
			if (br_out and tr_out):
				self.offcourt(player=1)

		else:
			#shrink both paddles, so they can't get the ball once it goes past
			playerleft.rect.inflate(-3,-3)
			playerright.rect.inflace(-3,-3)

			if self.rect.collidepoint(player1.rect) ==1 and not self.hitpaddle:
				self.hitpaddle = 1
				angle = math.pi - angle

			if self.rect.collidepoint(player2.rect) == 1 and not self.hitpaddle:
				self.hitpaddle = 1
				angle = math.pi - angle

			elif self.hitpaddle:
				self.hitpaddle = 0
		self.vector = (angle, vel)



#hiagain



class Paddle(pygame.sprite.Sprite):
	def _init_(self,side,speedb):
		self.side = side
		pygame.sprite.Sprite._init_(self)
		self.area = screen.get_rect
		if side == "left":
			self.paddle = pygame.Surface(self.area.midleft)
		elif side == "right":
			self.paddle = pygame.Surface(self.area.midright)
		
		paddle.fill((255,0,0))

		self.rect = self.paddle.get_rect()
		screen = pygame.display.get_surface()
		
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



def main():
	pygame.init()
	size = width,height = 600,400
	screen = pygames.display.set_mode(size)
	#Set Background
	background = pygame.Surface(screen.get_size())
	background = background.convert()
	background.fill((200,150,120))


	global playerleft
	playerleft = Paddle(side="left",speedb = 10)
	global playerright
	playerright = Paddle(side="right",speedb = 10)
	angle_vel = [23, 4]
	ball = Ball(angle_vel)

	playersprites = pygame.sprite.Renderplain((playerleft,playerright))
	ballsprite = pygrame.sprite.Renderplain((ball))

	screen.blit(background,(0,0))
	pygame.display.flip()

	clock = pygame.time.Clock()


	while 1:
		clock.tick(60)

		for event in pygame.event.get():
	    	if event.type == pygame.QUIT:
	    		return
	 		elif event.type == KEYDOWN:
	 			if event.key == K_a:
					playerleft.moveup()
				if event.key == K_z:
					playerleft.movedown()
				if event.key == K_UP:
					playerright.moveup()
				if event.key == K_DOWN:
					playerright.movedown()
	 		elif event.type == KEYUP:
	 			if (event.key == K_a or event.key == K_z):
	 				playerleft.state = "still"
	 				playerleft.moveby=[0,0]
	 			if (event.key == K_UP or event.key == K_DOWN):
	 				playerright.state = "still"
	 				playerright.moveby = [0,0]

	 	screen.blit(background, ball.rect, ball.rect)
		screen.blit(background, playerleft.rect, playerleft.rect)
		screen.blit(background, playerleft.rect, playerleft.rect)
		ballsprite.update()
		playersprites.update()
		ballsprite.draw(screen)
		playersprites.draw(screen)
        pygame.display.flip()

if _name_ == '_main_':main()

