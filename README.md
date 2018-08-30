***
# AlphaZero-version AI Program in the Game of Renju(Gomoku)

Introduction
====
Google DeepMind's recent AlphaZero algorithm, accepting no domain-specific knowledges, mastered the game of Go, chess and shogi through self-played reinforcement learning from scratch, showing the potential of general AI, which is the holy grail of the entire AI community. In this project, we implement an ***AlphaZero-version Renju(Gomoku) AI program*** and provide a ***step-by-step tutorial*** on the technique with [***English version***](https://github.com/PolyKen/AlphaRenju_Zero/blob/master/tutorial/gomoku_paper.pdf) and [***中文版***](https://github.com/PolyKen/AlphaRenju_Zero/blob/master/tutorial/gomoku_paper_chinese.pdf). You can also get access to our presentation [***ppt***](https://github.com/PolyKen/AlphaRenju_Zero/blob/master/tutorial/Gomoku%20PPT.pptx).

Future Works
====
Extend the AI to ***15 by 15*** free style Gomoku.

Experiment
====
Animation (8 by 8 board)
-------
<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/gif/ai_self_play.gif" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/gif/human(black)_vs_ai(white).gif" width="350px" height="350px"/>
</p>

Human vs AI (8 by 8 board)
-------
AI adopts deterministic policy with 1600 simulations per move.
<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/man_vs_ai_1.png" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/man_vs_ai_2.png" width="350px" height="350px"/>
</p>

<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/man_vs_ai_3.png" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/man_vs_ai_4.png" width="350px" height="350px"/>
</p>

<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/man_vs_ai_5.png" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/man_vs_ai_6.png" width="350px" height="350px"/>
</p>

AI vs AI (8 by 8 board)
-------
AI adopts semi-stochastic policy with 800 simulations per move.
<p class="half" align="center">
   <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/ai_vs_ai_1.png" width="350px" height="350px"/>
   <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/ai_vs_ai_2.png" width="350px" height="350px"/>
</p>

Loss Function (8 by 8 board)
-------
<p align="center">
   <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/loss_fun.jpg" width="350px" height="350px"/>
</p>

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
- ***Sun Yat-sen University***

Acknowledgement
-------
We would like to say thanks to ***BaiAn Chen*** from ***MIT*** and ***MingWen Liu*** from ***ShiningMidas Private Fund*** for their generous help throughout the research. We are also grateful to ***ZhiPeng Liang*** and ***Hao Chen*** from ***Sun Yat-sen University*** for their supports of the training process of our Gomoku AI. Without their supports, it's hard for us to finish such a complicated task.

Set up
====
Python Version
-------
- ***3.6***

Modules needed
-------
- ***tensorflow***
- ***keras***
- ***pygame***
- ***asyncio***
- ***numpy***
- ***matplotlib***
- ***easygui*** (optional)

Contact
====
- xiezh25@mail2.sysu.edu.cn
- fuxy28@mail2.sysu.edu.cn
- yujy25@mail2.sysu.edu.cn
