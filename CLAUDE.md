● Tennis Underdog Detection Project 

1 - The system should identify strong underdogs who are likely to win at least one set.

2 - Only ATP and WTA singles tournaments should be included in the system.

3 - Use machine learning models to improve the accuracy of underdog predictions, specifically targeting underdogs likely to win at least one set.

3 - Collect all data from the Odds API and feed it into the ML models.
https://the-odds-api.com/
Requests limits - 500 / Month

4 - Collect all data from https://www.tennisexplorer.com/ and feed it into the ML models.
Requests limits - 5 / Day

5 - Collect all data from 	--url https://tennisapi1.p.rapidapi.com/api/tennis/rankings/wta/live 
	                        --header 'x-rapidapi-host: tennisapi1.p.rapidapi.com' 
                          https://rapidapi.com/
Requests limits - 50 / Day
    
and also feed it into the ML models.
