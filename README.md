# Context-Skill Network in Flappy Ball environment

In order to deploy autonomous agents to domains such as autonomous driving, infrastructure management, health care, and finance, they must be able to adapt safely to unseen situations. The current approach in constructing such agents is to try to include as much variation into training as possible, and then generalize within the possible variations. This work proposes a principled approach where a context module (learns temporal variations) is coevolved with a skill module (learns spatial variations). The context module recognizes the variation and modulates the skill module so that the entire system performs well in unseen situations (i.e., spatio-temporal learning). The approach is evaluated in a challenging version of the Flappy Bird game where the effects of the actions vary over time. The Context+Skill approach leads to significantly more robust behavior in environments with previously unseen effects. Such a principled generalization ability is essential in deploying autonomous agents in real world tasks, and can serve as a foundation for continual learning as well.

Animations below show the test performance of all three evolved networks (i.e., network parameters are frozen after training) in a situation where the controlling parameters of the environment (i.e., effect of lift, gravity, etc.) are totally out of the training region.

Context-Skill Network
![Context-Skill Network Test Demo](demo/Context-Skill Net.gif) 

Context-only Network
![Context-only Network Test Demo](demo/Context-only Net.gif)

Skill-only Network
![Skill-only Network Test Demo](demo/Skill-only Net.gif)
