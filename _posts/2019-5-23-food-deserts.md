---
layout: page
title: Are food deserts in King County, WA very different from the rest of King County?
subtitle: We'll cross reference some demographic and fruit/vegetable consumption data from behavioral health surveys conducted by the Center for Health Statistics, a Division of Disease Control and Health Statistics at the Washington State Department of Health, with information on food deserts from the USDA.
image: /img/income_des.png
bigimg: /img/061616_best_farmers_markets.jpg
---

#### We'll look at the differences between all of King County and those living in food desert areas, do some basic analysis of two key indicators of inequality, and see whether people in food deserts report eating fresh foods less frequently.

First, let's examine the racial composition of food deserts[^1] in 
the county.

![Racial Composition of Respondents in Food Deserts](/img/race_des.png)

That may be surprising if you weren't aware that Seattle is in 
King County and that Seattle is one of the whitest places around.

![Racial Composition of All Respondents](/img/race_total.png)

Okay, so those look pretty much the same. Are they? Let's do some 
math and check with an individual t-test.

`pvalue = 0.9999`

If the groups were very different that `pvalue` would be less than 
0.01, but its almost 100 times that. This means that the racial 
makeup of King County as a whole is not significantly different 
from the makeup of food deserts in King County. They look pretty 
much the same because they are pretty much the same.

#### Alright, so what about income? Are food deserts composed of lower income earners?

![Income Composition of Respondents in Food Deserts](/img/income_des.png)

That is also pretty surprising, unless you're aware that the 
median income in King County in 2015 was a cool 
$[75k](https://www.kingcounty.gov/independent/forecasting/King%20County%20Economic%20Indicators/Household%20Income.aspx)!

![Income Composition of All Respondents](/img/income_total.png)

Just for a little perspective, according to MIT, the average 
expenses for a family of four in King County are about 
$[80k](http://livingwage.mit.edu/counties/53033).

These look the same, too. Are they? Let's do the math things!

`pvalue = 0.9999`

That `pvalue` is, again, too high to find any difference between 
those in, and those out of, food deserts.

#### Let's check out one more thing. This is where the rubber meets the road. Do people in King County food deserts eat fresh fruits and veggies less often?

![Consumption of Fruits and Veg](/img/consumption.png)

Most eat fruits and vegetables every day. That's great news! 

Okay, so what's the deal with this food desert thing anyway? 
I wouldn't necessarily think that over 80% of people in 
food deserts eat fruits and vegetables every day. Being in a 
food desert doesn't seem to make much of a difference. Is this even 
a thing? In short, yes. If you need further convincing please 
check out the links below. 

King County is atypical in many ways. Two obvious differences 
are the median income and cost of living. Another factor not 
represented in our data[^2]; public transportation is 
extensive throughout the area and is available on-demand and 
door-to-door for many vulnerable populations. Online ordering, 
door-to-door delivery services and neighborhood farmer's markets 
are also abundant. Also, biking and walking are encouraged with 
infrastructure improvements and public outreach. 

Access to affordable, healthy food options remains a challenge in 
many communities. Thankfully, getting fruits and vegetables in 
King County isn't too hard, even if you're over a mile from the 
store and live in a lower income neighborhood. 

It would be interesting to do similar research in other 
communities around the country and see if I am correct in
assuming that the USDA is right about food deserts being 
generally distinctive from their surrounding food oases.


[^1]: Food desert designations by the USDA can be found at the 
[Food Access Research Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas). 
Food deserts are defined as areas of 
[low income and low access](https://www.ers.usda.gov/webdocs/publications/93141/eib%20209%20summary.pdf?v=6737.3) 
and multiple designations exist based on the density of the 
population. King County is largly urban and all of our data 
pertains to the low income and 1 mile from nearest grocery store 
designation.

[^2]: Survey information can be found at the 
[Center for Health Statistics](https://www.doh.wa.gov/DataandStatisticalReports/DataSystems/BehavioralRiskFactorSurveillanceSystemBRFSS). 
The USDA food desert data is from 2015, so we only look at 
survey data from that year.

#####  You can see the code used for cleaning and organizing the data, and where to request your own copy [here](https://github.com/llpk79/Project_Blog_Post/blob/master/main.py).

##### Photo Credit: Arina P Habich/shutterstock
