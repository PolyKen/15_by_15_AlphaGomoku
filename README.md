***
# 15 by 15 AlphaGomoku

Introduction
====
- This is a Gomoku AI based on curriculum learning and AlphaGo methods.

<!---  (***2018-08-29***) We implement an ***AlphaGo-based Gomoku AI program*** in ***8 by 8 Free Style Gomoku***. You can also get access to our [***presentation PPT***](https://github.com/PolyKen/AlphaRenju_Zero/blob/master/tutorial/Gomoku%20PPT.pptx) in 2018 Likelihood Lab Summer Research Conference.
- (***2018-09-22***) We combine our original AlphaGomoku program with ***Curriculum Learning***, ***Double Networks Mechanism*** and ***Winning Value Decay*** to extend our AI to ***15 by 15 Free Style Gomoku***. Before we adopt these methods mentioned above, training 15 by 15 AlphaGomoku is intractable since the asymmetry and complexity of the game compared to the 8 by 8 simplified gomoku. 
- (***2018-9-25***) Our Reseach Paper is available at: [***paper***](https://github.com/PolyKen/15_by_15_AlphaGomoku/blob/master/tutorial/gomoku_paper.pdf)(or at [***arxiv***](http://arxiv.org/abs/1809.10595))
- The training is continuing...... We hope that AlphaGomoku can evolve into Gomoku grand master someday. -->


Demonstration
====
Human vs AlphaGomoku (15 by 15 board)
-------
AI adopts deterministic policy with 400 simulations per move. 
<p class="half" align="center">
  <img src="https://github.com/PolyKen/15_by_15_AlphaGomoku/blob/master/demo/picture/human_mcts_b1.PNG" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/15_by_15_AlphaGomoku/blob/master/demo/picture/human_mcts_b2.PNG" width="350px" height="350px"/>
</p>

<p class="half" align="center">
  <img src="https://github.com/PolyKen/15_by_15_AlphaGomoku/blob/master/demo/picture/human_mcts_b3.PNG" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/15_by_15_AlphaGomoku/blob/master/demo/picture/human_mcts_b4.PNG" width="350px" height="350px"/>
</p>

<!--
<p class="half" align="center">
  <img src="https://github.com/PolyKen/15_by_15_AlphaGomoku/blob/master/demo/picture/human_mcts_w1.PNG" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/15_by_15_AlphaGomoku/blob/master/demo/picture/human_mcts_w2.PNG" width="350px" height="350px"/>
</p>-->

<!--
<p class="half" align="center">
  <img src="https://github.com/PolyKen/15_by_15_AlphaGomoku/blob/master/demo/picture/human_mcts_w3.PNG" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/15_by_15_AlphaGomoku/blob/master/demo/picture/human_mcts_w4.PNG" width="350px" height="350px"/>
</p>-->


Tecent Gomoku AI(欢乐五子棋) vs AlphaGomoku (15 by 15 board)
-------
Tencent Gomoku AI plays black stone. AlphaGomoku adopts deterministic policy with 400 simulations per move.
<p align="center">
  <img src="https://github.com/PolyKen/15_by_15_AlphaGomoku/blob/master/demo/picture/tencent_1.jpg" width="350px" height="350px"/>
</p>

<!--
Animation (8 by 8 board)
-------
The left Gif is a game self played by AlphaGomoku; The right Gif is a game between human and ai, where human adopts balck stone. All AI simulate 400 times per move.
<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/gif/ai_self_play.gif" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/gif/human(black)_vs_ai(white).gif" width="350px" height="350px"/>
</p>-->

<!--
Human vs AlphaGomoku (8 by 8 board)
-------
AI plays the white stone against human, adopting deterministic policy with 400 simulations per move.
<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/man_vs_ai_1.png" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/man_vs_ai_2.png" width="350px" height="350px"/>
</p>
-->

Set up
====
Python Version
-------
- ***3.6***

Requirement
-------
`pip install -r requirements.txt`

- ***tensorflow***
- ***keras***
- ***pygame***
- ***numpy***
- ***matplotlib***
- ***easygui*** (optional)

How to play with AlphaGomoku
-------
- Execute run.py.
- Select mode 2 (AI vs Human).
- You can also compete with different versions of AlphaGomoku by switching the network.

Training
-------
- Execute run.py.
- Select mode 13.

Setting parameters
-------
All important parameters are in AlphaGomoku/config.py. Some of them are listed as follows,
- simulation_times: the number of 'exploration' of game tree for each move.
- c_puct: in general, when c_puct gets larger, the policy decision will rely more on prior probability.
- initial_tau: temperature coefficient. When it gets smaller, policy will tend to be more deterministic.

Contribution
====
Contributors
-------
- ***Zheng Xie***
- ***XingYu Fu***
- ***JinYuan Yu***

Institutions
-------
- ***Likelihood Lab***
- ***Vthree.AI***
- ***Sun Yat-sen University***

Acknowledgement
-------
We would like to say thanks to ***Andrew Chen*** from [***Vthree.AI***](http://vthree.ai/) and ***MingWen Liu*** from [***ShiningMidas Investment***](http://www.shiningmidas.com/) for their generous help throughout the research. We are also grateful to ***ZhiPeng Liang*** and ***Hao Chen*** from ***Sun Yat-sen University*** for their supports of the training process of our Gomoku AI. Without their supports, it's hard for us to finish such a complicated task.

Contact
====
- xiezh25@mail2.sysu.edu.cn
- fuxy28@mail2.sysu.edu.cn
- yujy25@mail2.sysu.edu.cn
