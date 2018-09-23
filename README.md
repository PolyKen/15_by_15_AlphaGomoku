***
# 15 by 15 AlphaGomoku

Introduction
====
-  (***2018-08-29***) We implement an ***AlphaGo version Gomoku AI program*** in ***8 by 8 Free Style Gomoku*** and provide a ***step-by-step tutorial*** on the technique with [***English version***](https://github.com/PolyKen/AlphaRenju_Zero/blob/master/tutorial/gomoku_paper.pdf) and [***中文版***](https://github.com/PolyKen/AlphaRenju_Zero/blob/master/tutorial/gomoku_paper_chinese.pdf). You can also get access to our [***presentation PPT***](https://github.com/PolyKen/AlphaRenju_Zero/blob/master/tutorial/Gomoku%20PPT.pptx) in 2018 Likelihood Lab Summer Research Conference.

- (***2018-09-22***) We combine our original AlphaGomoku program with ***Curriculum Learning***, ***Double Network Mechanism*** and ***Winning Value Decay*** to extend our AI to ***15 by 15 Free Style Gomoku***. Before we adopt these methods mentioned above, training 15 by 15 AlphaGomoku is intractable since the asymmetry and complexity of the game compared to the 8 by 8 simplified gomoku. 

Demonstration
====
Animation (15 by 15 board)
-------
Coming Soon.

Human vs AI (15 by 15 board)
-------
AI adopts semi-stochastic policy with 800 simulations per move. The first four pictures are games where AI plays the black stone. The following eight pictures are games where AI plays the white stone.
<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_black_1.PNG" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_black_2.PNG" width="350px" height="350px"/>
</p>

<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_black_3.PNG" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_black_4.PNG" width="350px" height="350px"/>
</p>

<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_white_1.PNG" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_white_2.PNG" width="350px" height="350px"/>
</p>

<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_white_3.PNG" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_white_4.PNG" width="350px" height="350px"/>
</p>

<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_white_5.PNG" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_white_6.PNG" width="350px" height="350px"/>
</p>

<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_white_7.PNG" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/15_ai_white_8.PNG" width="350px" height="350px"/>
</p>

Animation (8 by 8 board)
-------
The left Gif is a game self played by AlphaGomoku; The right Gif is a game between human and ai, where human adopts balck stone.
<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/gif/ai_self_play.gif" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/gif/human(black)_vs_ai(white).gif" width="350px" height="350px"/>
</p>

Human vs AI (8 by 8 board)
-------
AI plays the white stone against human, adopting deterministic policy with 1600 simulations per move.
<p class="half" align="center">
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/man_vs_ai_1.png" width="350px" height="350px"/>
  <img src="https://github.com/PolyKen/AlphaRenju_Zero/blob/master/demo/picture/man_vs_ai_2.png" width="350px" height="350px"/>
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
- ***threading***
- ***numpy***
- ***matplotlib***
- ***easygui*** (optional)

Contact
====
- xiezh25@mail2.sysu.edu.cn
- fuxy28@mail2.sysu.edu.cn
- yujy25@mail2.sysu.edu.cn
