import pygame 
import random
import numpy as np
import torch
from neural_net_torch import Context_Skill_Net as CSnet
# from neural_net import Context_Skill_Net as CSnet # Context-Skill Model
# from neural_net import Skill_Only_Net as S_o_net # Skill-only Model
# from neural_net import Context_Only_Net as C_o_net # Context-only Model
import subprocess
import pickle
from copy import deepcopy

#np.random.seed(123456)

#------------------------------------------------------------------------------

FPS = 30

SCREENWIDTH  = int(1.5*288)
SCREENHEIGHT = int(1.5*288)

WHITE = (255,255,255)
BLACK = (0,0,0)
RED   = (255,0,0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)
LIGHT_RED = (255, 153, 153)
LIGHT_BLUE = (102, 178, 255)

liftCount = 0
fwdFlapCount = 0

#------------------------------------------------------------------------------

font_name = pygame.font.match_font('arial')
text_size = 15
def printScore(surf, text, size, x, y):
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)


def saveFrame(screen, image_no):
    str_num = "000" + str(image_no)
    file_name = "image_test_" + str_num[-4:] + ".png"
    pygame.image.save(screen, file_name)

#------------------------------------------------------------------------------

"""
T1: Only GAP_POSITION changes, 2 parameters (lift=-15 or gravity=1) are normal
T2: In addition to GAP_POSITION, Lift changes
T3: In addition to GAP_POSITION, Gravity changes
T4: In addition to GAP_POSITION, Forward flap changes
T5: In addition to GAP_POSITION, Drag changes
"""

# NOTE: Use floating numbers for computation, but use integers for display.
    
class Bird():
    def __init__(self, net, game_params, params):
        lift     = params[1]
        gravity  = params[2]
        fwd_flap = params[3]
        drag     = params[4]

        self.y        = np.random.randint(2*SCREENHEIGHT/10, 8*SCREENHEIGHT/10) #int(SCREENHEIGHT / 2)
        self.x        = int(SCREENWIDTH * 0.2)
        self.rad      = game_params["BIRD_RADIUS"]
        # PARAMETERs ---------------------------------------------
        self.gravity  = gravity
        self.lift     = lift   
        self.fwd_flap = fwd_flap
        self.drag     = drag 
        # --------------------------------------------------------
        self.VelY     = -5
        self.MaxVelY  = 12  # max vel along Y, max descend speed
        self.MinVelY  = -12  # min vel along Y, max ascend speed
        self.VelX     = 5
        self.MaxVelX  = 12  # max vel along X
        self.MinVelX  = 3  # min vel along X
        # --------------------------------------------------------
        self.pipeCount = 0 # number of pipes passed successfully
        self.liftCount = 0
        self.descentCount = 0
        self.fwdFlapCount = 0
        self.slowDownCount = 0
        self.hitCount = 0
        self.offScreenCount = 0
        # --------------------------------------------------------
        # self.brain = deepcopy(net)
        self.brain = net
    
    def show(self, display, changeColor):
        if changeColor == False:
            pygame.draw.circle(display, RED, (round(self.x), round(self.y)), self.rad)
        else:
            pygame.draw.circle(display, LIGHT_RED, (round(self.x), round(self.y)), self.rad)
        
    def action(self, net_output, pipes=None):
        # the network has two output nodes
        if net_output[0,0] > 0.0: #UP (FLAP)
            self.VelY += self.lift
            self.liftCount += 1
        if net_output[0,1] > 0.0: #RIGHT (SPEED UP)
            self.VelX += self.fwd_flap
            self.fwdFlapCount += 1
    
    def clampVelY(self):
        if self.VelY > self.MaxVelY:
            self.VelY = self.MaxVelY
        elif self.VelY < self.MinVelY:
            self.VelY = self.MinVelY
            
    def clampVelX(self):
        if self.VelX > self.MaxVelX:
            self.VelX = self.MaxVelX
        elif self.VelX < self.MinVelX:
            self.VelX = self.MinVelX
        
    def update(self):
        self.VelY += self.gravity
        self.clampVelY()
        self.y += self.VelY
        self.VelX -= self.drag
        self.clampVelX()
        
    def updatePipeScore(self, pipes):
        for pipe in pipes:
            if (self.x-self.rad > (pipe.x+pipe.width)) and (pipe.passed == False):
                pipe.passed = True
                if (pipe.hit_before == False):
                    self.pipeCount += 1
        return pipes  
        
    def predict(self, pipes, game_params):
        # Find the closest pipe
        # 6-inputs:
        # ---------
        # 0 - bird.y
        # 1 - bird.VelY
        # 2 - bird.VelX
        # 3 - X-distance to the closest pipe
        # 4 - Height of the top pipe
        # 5 - Height of the bottom pipe
        closestPipe = None
        closestPipeDistX = 1e8
        if len(pipes) == 0:
            obs = np.zeros((1,6))
        else:
            for i in range(len(pipes)):
                d = (pipes[i].x + pipes[i].width) - self.x # Distance-X
                if (d < closestPipeDistX) and (d >= 0.0): # d > 0.0 for not to consider pipes behind
                    closestPipe = pipes[i]
                    closestPipeDistX = d
            obs = np.array([self.y / SCREENHEIGHT,
                            (self.VelY - self.MinVelY) / (self.MaxVelY - self.MinVelY),
                            (self.VelX - self.MinVelX) / (self.MaxVelX - self.MinVelX),
                            closestPipeDistX / SCREENWIDTH,
                            closestPipe.top / SCREENHEIGHT,
                            closestPipe.bottom / SCREENHEIGHT]).reshape((1,-1)) 
            obs_t = torch.from_numpy(obs)
            output = self.brain.forward(obs_t)
            self.action(output.detach().numpy(), pipes)
#        return (obs, output)
            
    def offScreen(self):
        return ((self.y + self.rad) > SCREENHEIGHT or (self.y - self.rad < 0))
            
            
#------------------------------------------------------------------------------
        
class Pipe():
    def __init__(self, game_params, bird):
        if 3 in game_params["Active_Tasks"]: # PIPE_GAP
            self.gap = np.random.randint(game_params["Param_Range"][3][0], game_params["Param_Range"][3][1]+1)
            self.top = int(SCREENHEIGHT/2) - int(self.gap/2)
            if 4 in game_params["Active_Tasks"]: # PIPE_GAP_POSITION
                self.gap_center = np.random.randint(game_params["Param_Range"][4][0], game_params["Param_Range"][4][1]+1)
                self.top = self.gap_center - int(self.gap/2)
        else:
            self.gap = game_params["PIPE_GAP"]
            if 4 in game_params["Active_Tasks"]: # PIPE_GAP_POSITION
                self.gap_center = np.random.randint(game_params["Param_Range"][4][0], game_params["Param_Range"][4][1]+1)
                self.top = self.gap_center - int(self.gap/2)
            else: # Neither Task-3 and 4 are active
                self.top = int(SCREENHEIGHT/2) - int(self.gap/2)
        self.bottom = SCREENHEIGHT - (self.top + self.gap)
        self.x = SCREENWIDTH
        self.VelX = bird.VelX
        self.hit_before = False
        
        if 2 in game_params["Active_Tasks"]: # PIPE_WIDTH
            self.width = np.random.randint(game_params["Param_Range"][2][0], game_params["Param_Range"][2][1]+1)
        else:
            self.width = game_params["PIPE_WIDTH"]
            
        self.passed = False
        self.hit_before = False
    
    
    def hits(self, bird):
        rectX = [self.x, self.x]; rectY = [0.0, self.top+self.gap]
        rectW = [self.width, self.width]; rectH = [self.top, self.bottom]
        
        cx = bird.x; cy = bird.y
        for p in range(2):
            rx = rectX[p]; ry = rectY[p]
            rw = rectW[p]; rh = rectH[p]
            testX = cx; testY = cy
            # First find the closest edge
            if (cx < rx):
                testX = rx # left edge
            elif (cx > rx+rw):
                testX = rx+rw # right edge            
            if (cy < ry):
                testY = ry # top edge (not possible)
            elif (cy > ry+rh):
                testY = ry+rh # bottom edge
            # Then check the Euclidean distance
            distX = cx - testX
            distY = cy - testY
            distance = np.sqrt(distX**2 + distY**2)
            if (distance < bird.rad - 1e-3):
                return True
        return False
    
    def show(self, display):
        pygame.draw.rect(display, WHITE, (round(self.x), 0, self.width, self.top), 0)  
        pygame.draw.rect(display, WHITE, (round(self.x), self.top+self.gap, self.width, self.bottom), 0)
    
    def update(self, bird):
        self.x -= bird.VelX
        
    def offScreen(self):
        if (self.x + self.width <= 0):
            return True
        else:
            return False
        
#------------------------------------------------------------------------------        
            
def Game(net, game_params, params, graphics=False, video=False):
    # seed     = params[0]
    # lift     = params[1]
    # gravity  = params[2]
    # flap_fwd = params[3]
    # drag     = params[4]
    pygame.init()
    if(graphics):
        display = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    FPSCLOCK = pygame.time.Clock()
    seed = params[0]
    np.random.seed(seed)
    random.seed(seed)
    #-------------------------
    agent = Bird(net, game_params, params)
    #-------------------------
    timeCount = 0 # time counter
    pipes = []
    distance = 0
    done = False
    start = True
    changeColor = False
    changeBackground = False
    while not done:
        if start:
            pipes.append(Pipe(game_params, agent))
            if 1 in game_params["Active_Tasks"]: # PIPE_SPACING
                spacing = np.random.randint(game_params["Param_Range"][1][0], game_params["Param_Range"][1][1]+1)
            else:
                spacing = game_params["PIPE_SPACING"]
            start = False
        elif SCREENWIDTH - (pipes[-1].x + pipes[-1].width) >= spacing:
            pipes.append(Pipe(game_params, agent))
            if 1 in game_params["Active_Tasks"]: # PIPE_SPACING
                spacing = np.random.randint(game_params["Param_Range"][1][0], game_params["Param_Range"][1][1]+1)
            else:
                spacing = game_params["PIPE_SPACING"]
            start = False
        timeCount += 1
        distance += agent.VelX
                
        if agent.offScreen():
            agent.y -= agent.VelY
            agent.offScreenCount += 1
            agent.hitCount += 5
            for pipe in pipes:
                if pipe.hit_before == False:
                    pipe.hit_before = True
            changeBackground = True
        
        # -------------------------------------
        # Manual control for debugging
#            events = pygame.event.get()
#            for event in events:
#                if event.type == pygame.KEYDOWN:
#                    if event.key == pygame.K_UP:
#                        agent.action(0)
#                    elif event.key == pygame.K_RIGHT:
#                        for pipe in pipes:
#                            agent.action(2, pipes)
#                    elif event.key == pygame.K_LEFT:
#                        for pipe in pipes:
#                            agent.action(3)
        # -------------------------------------
        agent.predict(pipes, game_params)
        agent.update()
        
        for pipe in pipes:
            pipe.update(agent) # Update pipe's position of the pipe using "bird.VelX"
            if pipe.hits(agent):
                agent.hitCount += 1
                if pipe.hit_before == False:
                    pipe.hit_before = True
                changeColor = True
            if pipe.offScreen():
                pipes.remove(pipe)
        
        pipes = agent.updatePipeScore(pipes)
        
        if(graphics):
            if changeBackground == False:
                display.fill(BLUE)
            else:
                display.fill(LIGHT_BLUE)
                changeBackground = False
            # Draw pipes
            for pipe in pipes:
                pipe.show(display)
            # Draw agent    
            agent.show(display, changeColor)
            changeColor = False
            # Print Information
            printScore(display, "Time: %d" % timeCount, text_size, int(SCREENWIDTH-50), 2*text_size)
#                printScore(display, "Pipes: %d" % agent.score, text_size, int(SCREENWIDTH/2), int(SCREENHEIGHT/2))
#                printScore(display, "Hits: %d" % hitCount, text_size, int(SCREENWIDTH/2), int(SCREENHEIGHT/2)+text_size)
#                printScore(display, "Offscreen: %d" % offScreenCount, text_size, int(SCREENWIDTH/2), int(SCREENHEIGHT/2)+2*text_size)
#                printScore(display, "%s" % '- - - - - - - - - - - -', text_size, int(SCREENWIDTH/2), int(SCREENHEIGHT/2)+3*text_size)
#                printScore(display, "Lift: %.1f" % agent.lift, text_size, int(SCREENWIDTH/2), int(SCREENHEIGHT/2)+4*text_size)
#                printScore(display, "Gravity: %.1f" % agent.gravity, text_size, int(SCREENWIDTH/2), int(SCREENHEIGHT/2)+5*text_size)
#                printScore(display, "Drag: %.1f" % agent.drag, text_size, int(SCREENWIDTH/2), int(SCREENHEIGHT/2)+6*text_size)
#                printScore(display, "Flap-forward: %.1f" % agent.fwd_flap, text_size, int(SCREENWIDTH/2), int(SCREENHEIGHT/2)+7*text_size)
#                printScore(display, "Bird-VelX: %.1f" % agent.VelX, text_size, int(SCREENWIDTH/2), int(SCREENHEIGHT/2)+8*text_size)
            printScore(display, "L= %.1f, G= %.1f, F-fwd= %.1f, D= %.1f" % (agent.lift, agent.gravity, agent.fwd_flap, agent.drag), int(0.9*text_size), 9*text_size, SCREENHEIGHT-2*text_size)
            printScore(display, "Pipes= %d, Hits= %d, Out= %d" % (agent.pipeCount, agent.hitCount, agent.offScreenCount), text_size, 7.5*text_size, 2*text_size)                
            
            
            pygame.display.update()
            FPSCLOCK.tick(FPS)
            saveFrame(display, timeCount)
            
        if timeCount >= game_params["MAX_TIME"]:
            done = True
    
    
    if video:
        if agent.brain.name == "Context-Skill":
            filename = "CS_"
        elif agent.brain.name == "Context-only":
            filename = "C_"
        if agent.brain.name == "Skill-only":
            filename = "S_"
        filename += "L%.1f_G%.1f_Fwd%.1f_D%.1f.mp4" % (params[1],params[2],params[3],params[4])
        command1 = "ffmpeg -framerate " + str(FPS) + " -r " + str(FPS) +\
                   " -s " + str(SCREENWIDTH) + "x" + str(SCREENHEIGHT) + \
                   " -i image_test_%04d.png -c:v libx264 -crf 10" +\
                   " -pix_fmt yuv420p " + filename
        command2 = "rm image_*"
        subprocess.call(command1, shell=True)
        subprocess.call(command2, shell=True)
    if graphics == True and video == False:
        command2 = "rm image_*"
        subprocess.call(command2, shell=True)
    #-------------------------
    # reward = pipeCount - 3.0*offScreenCount - 2.0*hitCount
    fit0 = agent.pipeCount
    fit1 = agent.hitCount
#    print("\nPipes: %d, Hits: %d, Outs: %d" % (pipeCount, hitCount, offScreenCount))
#    pickle.dump( np.array(obs_action_archive), open( "obs_and_actions.p", "wb" ) )
    return fit0, fit1

#------------------------------------------------------------------------------ 

if (__name__ == '__main__'):
    net = pickle.load( open("net_CS.p", "rb") )
    
    base_lift, base_gravity, base_fwd_flap, base_drag = -12., 1., 5., 1. 
    percent = 0.2
    
    game_params = {"MAX_TIME"     : 500, # Game will end after MAX_TIME
                   "PIPE_SPACING" : 160, # Fixed for now, but it can be variable
                   "PIPE_GAP"     : 130, # 130Fixed for now, but it can be variable
                   "PIPE_WIDTH"   : 60, # 60Fixed for now, but it can be variable
                   "PIPE_GAP_POS" : 216, # Start of the gap
                   "LIFT"         : base_lift,
                   "GRAVITY"      : base_gravity,
                   "FWD_FLAP"     : base_fwd_flap,
                   "DRAG"         : base_drag,
                   "BIRD_RADIUS"  : 14,
                   "Active_Tasks" : [4],    
                   "Param_Range"  : [[0,    0],     # 0) dummy variable
                                     [120,  200],   # 1) PIPE_SPACING 
                                     [20,   150],   # 2) PIPE_WIDTH
                                     [90,   170],   # 3) PIPE_GAP
                                     [100,  332],   # 4) PIPE GAP POSITION
                                     [base_lift * (1+percent), base_lift * (1-percent)],  # 5) BIRD_Lift
                                     [base_gravity * (1-percent), base_gravity * (1+percent)],  # 6) GRAVITY
                                     [base_fwd_flap * (1-percent), base_fwd_flap * (1+percent)],  # 7) BIRD_Forward_Flap
                                     [base_drag * (1-percent), base_drag * (1+percent)],  # 8) DRAG
                                     [10.0, 18.0]]} # 9) BIRD_RADIUS  


    seed, lift, gravity, fwd_flap, drag = 0, -12, 1., 5., 1.
    params = [seed, lift*1.5, gravity*0.5, fwd_flap*1.5, drag*1.5]
    fit0, fit1 = Game(net, game_params, params, graphics=True, video=True)
    print("pipeCount= %d, hitCount= %d\n\n" % (fit0, fit1))

    pygame.display.quit()
    pygame.quit()
    
    
    
    
    
    
    
    
