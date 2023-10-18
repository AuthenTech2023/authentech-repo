import pynput
from pynput import mouse
import pygame
import sys
import time
global start
global start_location
global start_switch
global prev_x
global m_end
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SECS = 480
SUBJECT_ID = 2
SUBJECT_GENDER = 0                                                                                                      # M = 0, F = 1
global count
global drag


def on_move(x, y):
	global prev_x
	global m_end
	global count
	global start_switch
	global start_location
	if start_switch == 0:
		start_location = (x, y)
		start_switch += 1
		prev_x = start_location[0]
		m_end = time.time() - 1e-2
	speed = (abs(prev_x - x))/(time.time() - m_end)
	acc = (time.time() - m_end)/(speed + 1e-5)
	print(time.time(), ';', x, ';', y, ';', -1, ';', -1, ';', start_location[0] - x, ';', start_location[1] - y, ';', speed, ';', acc, ';',
	      SUBJECT_GENDER, ';', SUBJECT_ID)
	prev_x = x
	m_end = time.time()
	listener.stop()



def on_click(x, y, button, pressed):
	global start
	global drag
	if str(button) == "Button.left":
		button = 0
	elif str(button) == "Button.right":
		button = 2
	else:
		button = 1
	end = time.time()
	formatted_time = "{:.5f}".format(end - start)
	if pressed:
		drag = time.time()
		print(time.time(), ';', x, ';', y, ';', button, ';', formatted_time, ';', start_location[0] - x, ';',
		      start_location[1] - y, ';', -1, ';', SUBJECT_GENDER, ';', SUBJECT_ID)
	if not pressed:
		if button == 0:
			release = 5
		elif button == 2:
			release = 6
		else:
			release = 7
		try:
			formatted_time = "{:.5f}".format(time.time() - drag)
			print(time.time(), ';', x, ';', y, ';', release, ';', formatted_time, ';', start_location[0] - x, ';',
			      start_location[1] - y, ';', -1, ';',  SUBJECT_GENDER, ';', SUBJECT_ID)
		except NameError:
			click = time.time()
			print(time.time(), ';', x, ';', y, ';', button, ';', formatted_time, ';', start_location[0] - x, ';',
			      start_location[1] - y, ';', -1, ';',  SUBJECT_GENDER, ';', SUBJECT_ID)
			formatted_time = "{:.5f}".format(time.time() - click)
			print(time.time(), ';', x, ';', y, ';', release, ';', formatted_time, ';', start_location[0] - x, ';',
			      start_location[1] - y, ';', -1, ';',  SUBJECT_GENDER, ';', SUBJECT_ID)
	start = time.time()
	listener.stop()


def on_scroll(x, y, dx, dy):
	if dy > 0:
		print(time.time(), ' ;', x, ';', y, ';', 4, ';', -1, ';', -1, ';', -1, ';', -1, ';',  SUBJECT_GENDER, ';', SUBJECT_ID)
	else:
		print(time.time(), ';', x, ';', y, ';', 3, ';', -1, ';', -1, ';', -1, ';', -1, ';',  SUBJECT_GENDER, ';', SUBJECT_ID)
	listener.stop()



def start_screen():
	running = True
	pygame.init()
	while running:
		screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
		font = pygame.font.SysFont('Arial', 25)
		text = font.render('Click \'START\' to begin', True, (255, 255, 255), (0, 0, 0))
		text2 = font.render('START', True, (255, 255, 255), (0, 0, 0))
		textRect2 = text2.get_rect()
		textRect = text.get_rect()
		textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
		textRect2.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 1.5)
		screen.fill((0, 0, 0))
		screen.blit(text, textRect)
		screen.blit(text2, textRect2)
		pygame.display.update()
		event = pygame.event.wait()
		if event.type == pygame.QUIT:
			running = False
			pygame.quit()
		elif event.type == pygame.MOUSEBUTTONDOWN:
			running = False
			pygame.quit()


if __name__ == '__main__':
	count = 0
	start_screen()
	# sys.stdout = open(f"Subject{SUBJECT_ID}.txt", "w")
	start_switch = 0
	start = time.time()
	now = time.time()
	end = now + SECS
	flag = True
	while time.time() < end:
		with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
			listener.join()
	listener.stop()
	# sys.stdout.close()