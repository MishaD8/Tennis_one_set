● Tennis Underdog Detection Project 

1 - The system should identify strong underdogs who are likely to win SECOND set.

2 - Only ATP and WTA singles tournaments should be included in the system.

3 - For our models, use only best-of-3 sets format as in ATP tournaments. Grand Slam events like Australian Open, French Open, Wimbledon, and US Open use best-of-5 sets but we will exclude those formats from our analysis.

4 - FOCUSING ONLY on ranks 101-300

5 - Use machine learning models to improve the accuracy of underdog predictions, specifically targeting underdogs likely to win SECOND SET !!!

6 - Collect all data from the Odds API and feed it into the ML models.
https://the-odds-api.com/
Requests limits - 500 / Month

7 - Collect all data from https://www.tennisexplorer.com/ and feed it into the ML models.
Requests limits - 5 / Day

8 - Collect all data from 	--url https://tennisapi1.p.rapidapi.com/api/tennis/rankings/wta/live 
	                        --header 'x-rapidapi-host: tennisapi1.p.rapidapi.com' 
                          https://rapidapi.com/
Requests limits - 50 / Day
    
and also feed it into the ML models.
