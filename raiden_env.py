import pygame
import random
import math
import numpy as np
import os
import skimage
from skimage.transform import resize
from skimage.color import rgb2gray
# os.environ["SDL_VIDEODRIVER"] = "x11"
# Global var
SPEED = 60
root = r'resource'
player = None
en_bullet1_group = None
p_bullet_group = None
time = None
btime = None
clock = None
allSprites = None
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
L_GREEN = (181, 230, 29)
D_GREEN = (0, 183, 0)
N_BLUE = (0, 128, 255)
ACTIONS = 9
pygame.init()
# Used to manage how fast the screen updates


# Set the width and height of the screen [width, height]
size = (700, 900)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("raiden")  # set the title of the game to alpha

# load background
x_speed = 0
y_speed = 0

x_coord = 350  # initial value of player coordinate x
y_coord = 800  # initial value of player coordinate y
step_length = 5
myscore = 0  # set the initial value of player score to zero

en_x_speed = 3
en_y_speed = 3
en_x_dir = 0
en_y_dir = 0
playerdead = False  # Boolean value for player state
# b1 = "background2.jpg"  # background image name
game_start = False
game_end = False
enemy_pic = "a-01.png"  # default image for enemy

# --- background music
##pygame.mixer.music.load("level1.mp3")  # play music as soon as the game start
##pygame.mixer.music.play(-1)  # loop around if music ends


# --- background image
# back = pygame.image.load(root + b1).convert()
# back2 = pygame.image.load(root + b1).convert()


##class background(pygame.sprite.Sprite):
##    def __init__(self):
##        pygame.sprite.Sprite.__init__(self)
##        self.image = pygame.image.load(root + "background.png").convert()
##        self.rect = self.image.get_rect()
##
##    def update_Up(self):
##        self.rect.y+=1

# --- Classes
class Player(pygame.sprite.Sprite):
    def __init__(self, shoot=False):
        pygame.sprite.Sprite.__init__(self)
        # self.image=pygame.Surface([width,height])
        # self.image.fill(color)
        self.image = pygame.image.load(os.path.join(root, "p02.png")).convert_alpha()  # set the player icon
        self.image.set_colorkey(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x_coord
        self.rect.y = y_coord
        self.x_speed = x_speed
        self.y_speed = y_speed
        # self.direct = direct
        self.pos = (self.rect.centerx, self.rect.centery)
        self.score = 0
        self.hp = 100
        self.live = 3
        self.playershoot = shoot
        self.count = 100
        self.n = True
        self.action = np.zeros([ACTIONS])
        # pygame.draw.ellipse(self.image, WHITE, [-28,-33, 90, 90])
        # self.image.set_colorkey(WHITE)
    def setAction(self, action):
        self.action = action

    def __getAction__(self):
        return self.action

    def shoot(self):
        self.count += 1
        if self.playershoot and self.count >= 8:
            p_bullet = Bullet()
            if self.n:
                p_bullet.rect.x = self.rect.centerx + 6
                self.n = False
            else:
                p_bullet.rect.x = self.rect.centerx - 19
                self.n = True
            p_bullet.rect.y = self.rect.centery - 18
            p_bullet_group.add(p_bullet)
            allSprites.add(p_bullet)
            self.count = 0

    def moveX(self, x_speed):
        if x_speed != 0:
            self.checkcollision(x_speed, self.y_speed)

    def moveY(self, y_speed):
        if y_speed != 0:
            self.checkcollision(self.x_speed, y_speed)

    def checkcollision(self, x_speed, y_speed):
        self.rect.x += x_speed
        self.rect.y += y_speed

    ##        for bullet in en_bullet_list:
    ##            if self.rect.colliderect(bullet.rect):
    ##                self.hp -= bullet.damage
    ##                if self.hp <= 0 and self.live < 0:
    ##                    playdead = True
    ##                elif self.hp <= 0:
    ##                    self.live -= 1

    ##    def shoot(playershoot):
    ##        pressed = pygame.key.get_pressed()
    ##
    ##
    ##        if event.type == pygame.KEYUP:

    def update(self):
        self.pos = (self.rect.centerx, self.rect.centery)
        #action 0  1    2    3     4   5   6   7   8
        #       up down left right u_l u_r d_l d_r stay
        action = self.__getAction__()
        if action[0] == 1:
            self.moveY(-5)
        if action[1] == 1:
            self.moveY(5)
        if action[2] == 1:
            self.moveX(-5)
        if action[3] == 1:
            self.moveX(5)
        if action[4] == 1:
            self.moveY(-5)
            self.moveX(-5)
        if action[5] == 1:
            self.moveY(-5)
            self.moveX(5)
        if action[6] == 1:
            self.moveY(5)
            self.moveX(-5)
        if action[7] == 1:
            self.moveY(5)
            self.moveX(5)


class HitBox(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(os.path.join(root, "hitbox.png")).convert()
        self.image.set_colorkey(WHITE)
        ##        pygame.draw.ellipse(self.image, YELLOW, [0,0, 7, 7])
        self.rect = self.image.get_rect()

    def update(self):  # set its position to player
        self.rect.x = player.rect.centerx - 3
        self.rect.y = player.rect.centery - 18


class Bullet(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(os.path.join(root, "pb.png")).convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.x = x_coord
        self.rect.y = y_coord
        self.x_speed = 0
        self.y_speed = -12
        # self.dmg = 1
        self.dmg = 5
        self.levelup = 0

    def update(self):
        self.rect.x += self.x_speed
        self.rect.y += self.y_speed

        ##        if x_speed != 0:
        ##            self.checkcollision(x_speed,0)
        ##        if y_speed != 0:
        ##            self.checkcollision(0,y_speed)
        ##
        ##    def checkcollision(self,x_speed,y_speed):


##        self.levelup += 1

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        # self.image=pygame.Surface([width,height])
        # self.image.fill(color)
        self.image = pygame.image.load(os.path.join(root, enemy_pic)).convert_alpha()
        self.image.set_colorkey(None)
        self.rect = self.image.get_rect()
        self.x_offset = -6
        self.y_offset = -6
        self.score = 20
        self.turn = 3.1415926535898 * 50  # angle counter for circular path
        self.radius = 100
        self.hp = 0
        self.crash_dmg = 20
        self.angle = self.get_angle(player.pos)  # gun angle
        self.count = 500
        # self.rotate = 0
        # pygame.draw.ellipse(self.image, WHITE, [-28,-33, 90, 90])
        # self.image.set_colorkey(WHITE)

    ##enemy bullet chase
    def get_angle(self, player):
        x = player[0] - self.rect.centerx
        y = player[1] - self.rect.centery - 20
        self.angle = 135 - math.degrees(math.atan2(y, x))
        return self.angle

    ##enemy shoot function
    def shoot(self, gap):
        self.count += 1
        if self.count >= gap:
            self.angle = self.get_angle(player.pos)
            en_bullet1 = en_Bullet(self.rect.centerx, self.rect.centery, self.angle)
            en_bullet1_group.add(en_bullet1)
            allSprites.add(en_bullet1)
            self.count = 0

    ##enemy go vertical up
    def update_Down(self):
        self.rect.y -= self.y_offset

    ##enemy move acorss screen
    def update_A(self):
        self.rect.y += self.y_offset * 2
        self.rect.x += self.x_offset

    ##enemy moving like "Z"
    def update_Z(self):
        if self.rect.x < 1 or self.rect.x > 620:
            self.x_offset = self.x_offset * -1
        self.rect.x += self.x_offset
        self.rect.y += self.y_offset/2

    ##enemy moving 180 from left
    def update_180(self):
        if self.rect.y >= 150 and self.rect.x == 150:
            self.rect.y += self.y_offset
        elif self.rect.y < 150:
            self.turn += 1
            self.rect.x = self.radius * math.cos(self.turn / 50) + 250
            self.rect.y = self.radius * math.sin(self.turn / 50) + 150
        elif self.rect.y >= 150 and self.rect.x > 250:
            self.rect.y -= self.y_offset

    ##enemy moving 180 from right
    def update_180_2(self):
        if self.rect.y >= 150 and self.rect.x == 550:
            self.rect.y += self.y_offset
        elif self.rect.y < 150:
            # self.image = pygame.transform.rotate(pygame.image.load(root + enemy_pic).convert_alpha(), self.rotate)
            # self.rotate += 1.15
            self.turn += 1
            self.rect.x = self.radius * math.sin(self.turn / 50) + 450
            self.rect.y = self.radius * math.cos(self.turn / 50) + 150
        elif self.rect.y >= 150 and self.rect.x < 450:
            self.rect.y -= self.y_offset

    ##enemy 90 turn from left
    def update_90(self):
        x = 280
        y = 150
        if self.rect.y == y and self.rect.x < x:
            self.rect.x -= self.x_offset * 2
        elif self.rect.x < (x + self.radius - 1):
            self.turn += 2
            self.rect.x = self.radius * math.cos(self.turn / 50) + x
            self.rect.y = self.radius * math.sin(self.turn / 50) + y + self.radius
        elif self.rect.y >= y and self.rect.x >= (x + self.radius - 1):
            self.rect.y -= self.y_offset * 2

    ##enemy 90 turn from right
    def update_90_2(self):
        x = 420
        y = 150
        if self.rect.y == y and self.rect.x > x:
            self.rect.x += self.x_offset * 2
        elif self.rect.x > (x - self.radius + 1):
            self.turn += 2
            self.rect.x = self.radius * math.sin(self.turn / 50) + x
            self.rect.y = self.radius * math.cos(self.turn / 50) + y + self.radius
        elif self.rect.y >= y and self.rect.x <= (x - self.radius + 1):
            self.rect.y -= self.y_offset * 2

    ##enemy suicide attack
    def update_S(self):
        if self.rect.x > player.rect.x:
            self.rect.x += self.x_offset
        elif self.rect.x < player.rect.x:
            self.rect.x -= self.x_offset
        if self.rect.y < player.rect.y:
            self.rect.y -= self.y_offset
        elif self.rect.y > player.rect.y:
            self.rect.y += self.y_offset

    ##enemy turret
    def update_T(self):
        self.rect.y += 1
        if 1 < time % 200 < 100:
            self.shoot(3)

    ##enemy boss
    def update_boss(self):
        self.turn += 1
        self.rect.x = self.radius * math.cos(self.turn / 100) + 350
        self.rect.y = self.radius / 2 * math.sin(self.turn / 100) + 100
        if 1 < time % 200 < 120:
            self.shoot(5)


class en_Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, angle):
        pygame.sprite.Sprite.__init__(self)
        self.angle = -math.radians(angle - 135)
        self.image = pygame.image.load(os.path.join(root, "bullet3.png")).convert_alpha()
        self.rect = self.image.get_rect()
        self.move = [x, y]
        self.speed_magnitude = 5
        self.speed = (self.speed_magnitude * math.cos(self.angle),
                      self.speed_magnitude * math.sin(self.angle))
        self.done = False
        self.dmg = 5
        self.count = 0

    def update(self):
        self.move[0] += self.speed[0]
        self.move[1] += self.speed[1]
        self.rect.topleft = self.move

    def update_spin(self):
        self.count += 1
        self.speed = (self.speed_magnitude * math.cos(self.count),
                      self.speed_magnitude * math.sin(self.count))
        self.update()



# -------- Main Program Loop ------------------- Main Program Loop ------------------- Main Program Loop ----------

# game_start = True

class Raiden_Env():
    def __init__(self):
        pass

    # --- Enemy generation for testing
    def enemytest(self):
        global allSprites, enemytest_group, player, time, bgtime
        if time % 40 == 0:
            enemy = Enemy()
            enemy.rect.x = -20
            enemy.rect.y = 150
            enemy.turn = 3.1415926535898 * 75
            # enemy.turn = 3.1415926535898 * 50
            allSprites.add(enemy)
            enemytest_group.add(enemy)

    # --- Enemy move in 180 path
    def enemytype1(self):
        global allSprites, enemy1_group, time, bgtime
        if time % 40 == 0 and time % 4000 < 1000:
            enemy1 = Enemy()
            enemy1.hp = 3
            enemy1.angle = 3.1415926535898 * 25
            enemy1.rect.x = 550
            enemy1.rect.y = 900
            allSprites.add(enemy1)
            enemy1_group.add(enemy1)

    # --- Enemy suicide bomber
    def enemytype2(self):
        global allSprites, enemy2_group, player, time, bgtime
        if time % 30 == 0 and time % 4000 > 1000 and time % 4000 < 2000:
            enemy2 = Enemy()
            enemy2.hp = 3
            enemy2.crash_dmg = 40
            enemy2.y_offset = -5
            enemy2.rect.x = random.randint(10, 690)
            enemy2.rect.y = 0
            allSprites.add(enemy2)
            enemy2_group.add(enemy2)

    # --- Enemy with Z shape path
    def enemytype3(self):
        global allSprites, enemy3_group, time, bgtime
        if time % 10 == 0 and 1 < time % 300 < 250:
            enemy3 = Enemy()
            enemy3.image = pygame.image.load(os.path.join(root, "z-01.png")).convert_alpha()
            enemy3.hp = 4
            enemy3.rect.x = 0
            enemy3.rect.y = 900
            allSprites.add(enemy3)
            enemy3_group.add(enemy3)

    # --- Enemy move in 180
    def enemytype4(self):
        global allSprites, enemy4_group, time, bgtime
        if time % 40 == 0 and time % 4000 > 3000 and time % 4000 < 4000:
            enemy4 = Enemy()
            enemy4.hp = 3
            enemy4.rect.x = 150
            enemy4.rect.y = 900
            allSprites.add(enemy4)
            enemy4_group.add(enemy4)

    # --- Enemy combined, 2 group of enemy moving in 180 path
    def enemytype5(self):
        global p_bullet_group, en_bullet1_group, player, \
            time, bgtime, player_group, clock, allSprites, \
            hitbox, enemytest_group, enemy1_group, enemy2_group, \
            enemy3_group, enemy4_group, enemy5_group, enemy6_group, \
            enemyt_group, boss_group, myfont, instrucfont, hitbox_group
        if time % 50 == 0:
            enemy1 = Enemy()
            enemy1.image = pygame.image.load(os.path.join(root, "a-03.png")).convert_alpha()
            enemy1.hp = 4
            enemy1.turn = 3.1415926535898 * 25
            enemy1.rect.x = 550
            enemy1.rect.y = 900
            allSprites.add(enemy1)
            enemy1_group.add(enemy1)
        if (time + 25) % 50 == 0:
            enemy4 = Enemy()
            enemy4.image = pygame.image.load(os.path.join(root, "a-04.png")).convert_alpha()
            enemy4.hp = 4
            enemy4.rect.x = 150
            enemy4.rect.y = 900
            allSprites.add(enemy4)
            enemy4_group.add(enemy4)

    # --- suicide bomber with generation gap
    def enemytype6(self):
        global p_bullet_group, en_bullet1_group, player, \
            time, bgtime, player_group, clock, allSprites, \
            hitbox, enemytest_group, enemy1_group, enemy2_group, \
            enemy3_group, enemy4_group, enemy5_group, enemy6_group, \
            enemyt_group, boss_group, myfont, instrucfont, hitbox_group, enemy2

        if time % 10 == 0 and 0 < time % 300 < 250:
            enemy2 = Enemy()
            enemy2.image = pygame.image.load(os.path.join(root, "s-01.png")).convert_alpha()
            enemy2.hp = 1
            enemy2.crash_dmg = 40
            enemy2.y_offset = -5
            enemy2.rect.x = random.randint(10, 690)
            enemy2.rect.y = 0
            allSprites.add(enemy2)
            enemy2_group.add(enemy2)

    # --- combination of two 90 path enemy
    def enemytype7(self):
        global p_bullet_group, en_bullet1_group, player, \
            time, bgtime, player_group, clock, allSprites, \
            hitbox, enemytest_group, enemy1_group, enemy2_group, \
            enemy3_group, enemy4_group, enemy5_group, enemy6_group, \
            enemyt_group, boss_group, myfont, instrucfont, hitbox_group

        if time % 40 == 0 and 1 < time % 500 < 450:
            enemy5 = Enemy()
            enemy5.image = pygame.image.load(os.path.join(root, "a-01.png")).convert_alpha()
            enemy5.hp = 4
            enemy5.rect.x = -20
            enemy5.rect.y = 150
            enemy5.turn = 3.1415926535898 * 75
            allSprites.add(enemy5)
            enemy5_group.add(enemy5)
        if (time + 20) % 40 == 0 and 1 < time % 500 < 450:
            enemy6 = Enemy()
            enemy6.image = pygame.image.load(os.path.join(root, "a-02.png")).convert_alpha()
            enemy6.hp = 4
            enemy6.rect.x = 720
            enemy6.rect.y = 150
            enemy6.turn = 3.1415926535898 * 50
            allSprites.add(enemy6)
            enemy6_group.add(enemy6)

    def enemytype8(self):
        global p_bullet_group, en_bullet1_group, player, \
            time, bgtime, player_group, clock, allSprites, \
            hitbox, enemytest_group, enemy1_group, enemy2_group, \
            enemy3_group, enemy4_group, enemy5_group, enemy6_group, \
            enemyt_group, boss_group, myfont, instrucfont, hitbox_group
        if time % 30 == 0 and 1 < time % 600 < 500:
            enemy5 = Enemy()
            enemy5.image = pygame.image.load(os.path.join(root, "a-01.png")).convert_alpha()
            enemy5.hp = 4
            enemy5.rect.x = -20
            enemy5.rect.y = 150
            enemy5.turn = 3.1415926535898 * 75
            allSprites.add(enemy5)
            enemy5_group.add(enemy5)
        if time % 30 == 0 and 200 < time % 600 < 500:
            enemy6 = Enemy()
            enemy6.image = pygame.image.load(os.path.join(root, "a-02.png")).convert_alpha()
            enemy6.hp = 4
            enemy6.rect.x = 720
            enemy6.rect.y = 150
            enemy6.turn = 3.1415926535898 * 50
            allSprites.add(enemy6)
            enemy6_group.add(enemy6)

    # --- Enemy turret
    def enemytower(self):
        global enemyt
        global p_bullet_group, en_bullet1_group, player, \
            time, bgtime, player_group, clock, allSprites, \
            hitbox, enemytest_group, enemy1_group, enemy2_group, \
            enemy3_group, enemy4_group, enemy5_group, enemy6_group, \
            enemyt_group, boss_group, myfont, instrucfont, hitbox_group
        if time % 300 == 0:
            enemyt = Enemy()
            enemyt.image = pygame.image.load(os.path.join(root, "t-01.png")).convert_alpha()
            enemyt.hp = 10
            enemyt.crash_dmg = 50
            enemyt.score = 50
            enemyt.rect.x = random.randint(200, 500)
            enemyt.rect.y = 0
            allSprites.add(enemyt)
            enemyt_group.add(enemyt)

    # --- boss
    def enemyboss1(self):
        global p_bullet_group, en_bullet1_group, player, \
            time, bgtime, player_group, clock, allSprites, \
            hitbox, enemytest_group, enemy1_group, enemy2_group, \
            enemy3_group, enemy4_group, enemy5_group, enemy6_group, \
            enemyt_group, boss_group, myfont, instrucfont, hitbox_group
        boss1 = Enemy()
        boss1.score = 1000
        boss1.hp = 200
        boss1.crash_dmg = 100
        boss1.rect.x = 350
        boss1.rect.y = 100
        boss1.image = pygame.image.load(os.path.join(root, "boss1.png")).convert_alpha()
        boss1.rect = boss1.image.get_rect()
        allSprites.add(boss1)
        boss_group.add(boss1)

    def step(self, action):
        game_end = False
        global p_bullet_group, en_bullet1_group, player, \
            time, bgtime, player_group, clock, allSprites, \
            hitbox, enemytest_group, enemy1_group, enemy2_group, \
            enemy3_group, enemy4_group, enemy5_group, enemy6_group, \
            enemyt_group, boss_group, myfont, instrucfont, hitbox_group, SPEED
        # --- End screen with game reset
        # screen.fill(BLACK)
        old_score = player.score
        pygame.display.flip()

        # --- Main event loop
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        # --- Game logic should go here
        time += 1
        bgtime += 1

        # --- background reset
        if bgtime == 900:
            bgtime = 0
        # --- Level 1 process
        if time <= 1200:
            self.enemytype3()
        elif 1200 < time <= 2400:
            self.enemytype8()
        elif 2400 < time <= 3600:
            self.enemytype7()
        elif 3600 < time <= 4800:
            self.enemytype5()
        elif 4800 < time <= 6000:
            self.enemytype6()
        elif 6000 < time <= 7200:
            self.enemytower()
        elif time == 7401:
            self.enemyboss1()

        # --- Enemy Update
        for enemy in enemytest_group:
            enemy.update_90()
        for enemy in enemy1_group:
            enemy.update_180_2()
        for enemy in enemy2_group:
            enemy.update_S()
        for enemy in enemy3_group:
            enemy.update_Z()
        for enemy in enemy4_group:
            enemy.update_180()
        for enemy in enemy5_group:
            enemy.update_90()
        for enemy in enemy6_group:
            enemy.update_90_2()
        for enemy in enemyt_group:
            enemy.update_T()
        for boss in boss_group:
            boss.update_boss()

        # --- Player control limit
        if player.rect.x < 1:
            player.rect.x = 1
        elif player.rect.x > 640:
            player.rect.x = 640

        if player.rect.y < 1:
            player.rect.y = 1
        elif player.rect.y > 850:
            player.rect.y = 850

        # --- Player bullet generation code
        player.shoot()
        p_bullet = Bullet()

        # --- Remove bullet outside the screen
        for bullet in p_bullet_group:
            if bullet.rect.x > 725 or bullet.rect.x < -25 or bullet.rect.y < -25 or bullet.rect.y > 925:
                p_bullet_group.remove(bullet)
                allSprites.remove(bullet)

        for bullet in en_bullet1_group:
            if bullet.rect.x > 725 or bullet.rect.x < -25 or bullet.rect.y < -25 or bullet.rect.y > 925:
                en_bullet1_group.remove(bullet)
                allSprites.remove(bullet)

        # --- Player enemy collision
        enemy1_hit_player = pygame.sprite.groupcollide(hitbox_group, enemy1_group, False, True)
        enemy2_hit_player = pygame.sprite.groupcollide(hitbox_group, enemy2_group, False, True)
        enemy3_hit_player = pygame.sprite.groupcollide(hitbox_group, enemy3_group, False, True)
        enemy4_hit_player = pygame.sprite.groupcollide(hitbox_group, enemy4_group, False, True)
        enemy5_hit_player = pygame.sprite.groupcollide(hitbox_group, enemy5_group, False, True)
        enemy6_hit_player = pygame.sprite.groupcollide(hitbox_group, enemy6_group, False, True)
        enemyt_hit_player = pygame.sprite.groupcollide(hitbox_group, enemyt_group, False, True)
        boss_hit_player = pygame.sprite.groupcollide(hitbox_group, boss_group, False, False)

        for hitbox in enemy1_hit_player:
            player.hp -= enemy.crash_dmg
            player.score -= enemy.crash_dmg

        for hitbox in enemy2_hit_player:
            # player.hp -= enemy2.crash_dmg
            player.hp -= enemy.crash_dmg * 2
            player.score -= enemy.crash_dmg

        for hitbox in enemy3_hit_player:
            player.hp -= enemy.crash_dmg
            player.score -= enemy.crash_dmg

        for hitbox in enemy4_hit_player:
            player.hp -= enemy.crash_dmg
            player.score -= enemy.crash_dmg

        for hitbox in enemy5_hit_player:
            player.hp -= enemy.crash_dmg
            player.score -= enemy.crash_dmg

        for hitbox in enemy6_hit_player:
            player.hp -= enemy.crash_dmg
            player.score -= enemy.crash_dmg

        for hitbox in boss_hit_player:
            player.hp -= boss.crash_dmg
            player.score -= boss.crash_dmg

        for hitbox in enemyt_hit_player:
            # player.hp -= enemyt.crash_dmg
            player.hp -= enemy.crash_dmg
            player.score -= enemy.crash_dmg
        # --- Player's bullet enemy collision
        bullet_hit_enemy1 = pygame.sprite.groupcollide(enemy1_group, p_bullet_group, False, True)
        bullet_hit_enemy2 = pygame.sprite.groupcollide(enemy2_group, p_bullet_group, False, True)
        bullet_hit_enemy3 = pygame.sprite.groupcollide(enemy3_group, p_bullet_group, False, True)
        bullet_hit_enemy4 = pygame.sprite.groupcollide(enemy4_group, p_bullet_group, False, True)
        bullet_hit_enemy5 = pygame.sprite.groupcollide(enemy5_group, p_bullet_group, False, True)
        bullet_hit_enemy6 = pygame.sprite.groupcollide(enemy6_group, p_bullet_group, False, True)
        bullet_hit_enemyt = pygame.sprite.groupcollide(enemyt_group, p_bullet_group, False, True)
        bullet_hit_boss = pygame.sprite.groupcollide(boss_group, p_bullet_group, False, True)

        for enemy1 in bullet_hit_enemy1:
            enemy1.hp -= p_bullet.dmg
            if enemy1.hp <= 0:
                enemy1_group.remove(enemy1)
                allSprites.remove(enemy1)
                player.score += enemy1.score
            if enemy1.rect.x > 725 or enemy1.rect.x < -25 or enemy1.rect.y < -25 or enemy1.rect.y > 925:
                enemy1_group.remove(enemy1)
                allSprites.remove(enemy1)

        for enemy2 in bullet_hit_enemy2:
            enemy2.hp -= p_bullet.dmg
            if enemy2.hp <= 0:
                enemy2_group.remove(enemy2)
                allSprites.remove(enemy2)
                player.score += enemy2.score
            if enemy2.rect.x > 725 or enemy2.rect.x < -25 or enemy2.rect.y < -25 or enemy2.rect.y > 925:
                enemy2_group.remove(enemy2)
                allSprites.remove(enemy2)

        for enemy3 in bullet_hit_enemy3:
            enemy3.hp -= p_bullet.dmg
            if enemy3.hp <= 0:
                enemy3_group.remove(enemy3)
                allSprites.remove(enemy3)
                player.score += enemy3.score
            if enemy3.rect.x > 725 or enemy3.rect.x < -25 or enemy3.rect.y < -25 or enemy3.rect.y > 925:
                enemy3_group.remove(enemy3)
                allSprites.remove(enemy3)

        for enemy4 in bullet_hit_enemy4:
            enemy4.hp -= p_bullet.dmg
            if enemy4.hp <= 0:
                enemy4_group.remove(enemy4)
                allSprites.remove(enemy4)
                player.score += enemy4.score
            if enemy4.rect.x > 725 or enemy4.rect.x < -25 or enemy4.rect.y < -25 or enemy4.rect.y > 925:
                enemy4_group.remove(enemy4)
                allSprites.remove(enemy4)

        for enemy5 in bullet_hit_enemy5:
            enemy5.hp -= p_bullet.dmg
            if enemy5.hp <= 0:
                enemy5_group.remove(enemy5)
                allSprites.remove(enemy5)
                player.score += enemy5.score
            if enemy5.rect.x > 725 or enemy5.rect.x < -25 or enemy5.rect.y < -25 or enemy5.rect.y > 925:
                enemy5_group.remove(enemy5)
                allSprites.remove(enemy5)

        for enemy6 in bullet_hit_enemy6:
            enemy6.hp -= p_bullet.dmg
            if enemy6.hp <= 0:
                enemy6_group.remove(enemy6)
                allSprites.remove(enemy6)
                player.score += enemy6.score
            if enemy6.rect.x > 725 or enemy6.rect.x < -25 or enemy6.rect.y < -25 or enemy6.rect.y > 925:
                enemy6_group.remove(enemy6)
                allSprites.remove(enemy6)

        for enemyt in bullet_hit_enemyt:
            enemyt.hp -= p_bullet.dmg
            if enemyt.hp <= 0:
                enemyt_group.remove(enemyt)
                allSprites.remove(enemyt)
                player.score += enemyt.score
            if enemyt.rect.x > 725 or enemyt.rect.x < -25 or enemyt.rect.y < -25 or enemyt.rect.y > 925:
                enemy4_group.remove(enemyt)
                allSprites.remove(enemyt)

        for boss in bullet_hit_boss:
            boss.hp -= p_bullet.dmg
            if boss.hp <= 0:
                boss.kill()
                game_end = True
                player.score += boss.score

        # --- Enemy bullet and Player collision
        bullet_hit_player1 = pygame.sprite.groupcollide(hitbox_group, en_bullet1_group, False, True)

        for hitbox in bullet_hit_player1:
            player.hp -= 10
            player.score -= 10

        # --- Player damage checking
        if player.hp <= 0:
            player.live -= 1
            player.hp = 100
            # player.score -= 200
        ##        bullet.gap = 16

        if player.live <= 0:
            game_end = True

        # --- Blit screens
        # p_score = instrucfont.render('Score:' + str(player.score), 1, N_BLUE)
        # p_live = instrucfont.render('Live:' + str(player.live), 1, N_BLUE)
        # p_hp = instrucfont.render('HP:' + str(player.hp) + '/100', 1, N_BLUE)
        # --- Screen-clearing code goes here
        screen.fill(BLACK)

        # screen.blit(back, (0, bgtime))
        # screen.blit(back, (0, bgtime - 900))
        #
        # screen.blit(p_score, (15, 20))
        # screen.blit(p_hp, (300, 20))
        # screen.blit(p_live, (580, 20))
        # set action
        player.setAction(action)
        allSprites.update()
        allSprites.draw(screen)

        # --- Go ahead and update the screen with what we've drawn.
        #pygame.display.flip()
        # --- Limit to 60 frames per second
        clock.tick(SPEED)
        # return screenshot, reward, terminated_signal
        # add reward for live
        reward = player.score - old_score
        if not game_end and reward == 0:
            reward -= 1
        # if game_end:
        #     reward -= 100
        x_t = self.get_preprocessed_frame(pygame.surfarray.array3d(pygame.display.get_surface()))
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
        return s_t, \
               reward/100.0, \
               game_end, \
               player.score, \
               player.hp, \
               player.live

    def reset(self):
        global p_bullet_group, en_bullet1_group, player, \
            time, bgtime, player_group, clock, allSprites, \
            hitbox, enemytest_group, enemy1_group, enemy2_group, \
            enemy3_group, enemy4_group, enemy5_group, enemy6_group, \
            enemyt_group, boss_group, myfont, instrucfont, hitbox_group
        player_group = pygame.sprite.Group()
        clock = pygame.time.Clock()
        allSprites = pygame.sprite.Group()
        hitbox_group = pygame.sprite.Group()
        enemytest_group = pygame.sprite.Group()
        enemy1_group = pygame.sprite.Group()
        enemy2_group = pygame.sprite.Group()
        enemy3_group = pygame.sprite.Group()
        enemy4_group = pygame.sprite.Group()
        enemy5_group = pygame.sprite.Group()
        enemy6_group = pygame.sprite.Group()
        enemyt_group = pygame.sprite.Group()
        boss_group = pygame.sprite.Group()
        p_bullet_group = pygame.sprite.Group()
        en_bullet1_group = pygame.sprite.Group()
        # --- generation of player and hitbox
        hitbox = HitBox()
        hitbox_group.add(hitbox)
        allSprites.add(hitbox)
        player = Player(shoot=True)
        player_group.add(player)
        allSprites.add(player)# pygame.quit()
        myfont = pygame.font.SysFont('freesansbold.ttf', 80)
        instrucfont = pygame.font.SysFont('freesansbold.ttf', 50)
        time = 0
        bgtime = 0

    def get_preprocessed_frame(self, observation):
        """
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
        """
        return resize(rgb2gray(observation), (84, 84))

    def get_initial_state(self):
        self.reset()
        do_nothing = np.zeros([ACTIONS])
        do_nothing[8] = 0
        x_t, r_0, terminal, score, hp, live = self.step(do_nothing)
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
        # s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4
        return s_t
