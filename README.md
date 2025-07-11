# Chess Cheating Detection
 Chess engines (computer programs that play chess) are now so strong that anybody with a mobile phone can use it to cheat and defeat even the world's strongest players.  This repository aims to reproduce the statistical chess cheating screening software developed by Dr. Ken Regan and used by FIDE (although it is affiliated with neither.)  

## How it works 
 Suppose you want to test whether a coin is rigged.  You might try counting the number of heads you get when flipping the coin 100 times, and comparing that with what you'd expect to see from a fair coin.  For example, if you counted 51 heads, you'd likely conclude that the coin is fair (since it's very close to the 50 heads you'd expect to see from flipping a fair coin 100 times.)  If you counted 60 heads, you might be a little suspicious, but chalk it up to random chance.  However, if all 100 of your flips were heads, you'd likely conclude that the coin is rigged, since it'd be extraordinarily unlikely for such a result to happen by luck.  We can add some formality to this idea by introducing probability distributions, which tell us the exact probability of getting a certain result (e.g. 50 heads.)

 This program works on a similar idea to flag chess cheating.  A model is trained on thousands of human games to predict how an honest player might play any given position.  This model is then used to project a probability distribution of how an honest player would perform over the course of an entire game, where performance is quantified by several "aggregate statistics" (chief among which are "move match" and "aggregate error" as measured against an engine.)  To screen a specific player for cheating, we compute the aggregate statistics they achieved over a game (or many games.)  Then, we can use our model's projections to check the probability of an honest player achieving those aggregate statistics.  If the probability is extremely low (say p < 0.001), we might claim to have statistical evidence that the player cheated.

 Importantly, the model can project different performances for different games because it takes into account the specific positions in that game.  A very good aggregate error (you can think of this as similar to chess.com's accuracy metric) isn't necessarily suspicious if the opponent played terribly, while a less good aggregate error might nonetheless be suspicious if the game was incredibly complex.  

 Of course, the actual mechanics of this chess cheating detection program are a bit more complicated, so please check out my video to learn more!

### What this repo provides
 I've included 3000+ anonymized chess.com rapid games played by honest human players rated in the neighborhood of 2000 Elo.  I've also included Stockfish 17 NNUE's analysis of each game at depth = 12 and pv = 20 and the model parameters fit to this data.
 
