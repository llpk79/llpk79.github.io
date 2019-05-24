---
layout: post
title: Are food deserts in King County, WA very different from the rest of King County?
subtitle: We'll cross reference some demographic and fruit/vegetable consumption data from behavioral health surveys conducted by the [Center for Health Statistics](https://www.doh.wa.gov/DataandStatisticalReports/DataSystems/BehavioralRiskFactorSurvei ), a Division of Disease Control and Health Statistics at the Washington State Department of Health, with information on food deserts from the USDA.
image: /img/income_des.png
layout: page
---

##### Food desert designations by the USDA can be found at the [Food Access Research Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas).

##### Food deserts are defined as areas of [low income and low access](https://www.ers.usda.gov/webdocs/publications/93141/eib%20209%20summary.pdf?v=6737.3) and a few designations exist based on the density of the population.

##### King County is largly urban and all of our data pertains the low income and 1 mile from nearest grocery store designation.

##### The USDA food desert data is from 2015, so we'll just look at survey data from that year. We have 11,466 surveys to look through. We also have several columns we won't get to right now, but we have time take a look at a few that are typically looked at as indicators of inequality.

#####  You can see the process used for cleaning and organizing the data [here](https://github.com/llpk79/Project_Blog_Post/blob/master/main.py).

### We'll look at the differences between all of King County and just those living in food desert areas.

## Let's examine the racial composition of food deserts in the county.

![Racial Composition of Respondents in Food Deserts](/img/race_des.png)

### That may be surprising if you weren't aware that Seattle is in King County and that Seattle is one of the whitest places around.

![Racial Composition of All Respondents](/img/race_total.png)

### Told you so.

### Okay, so those look pretty much the same. Are they? Let's do some math and check with an individual t-test.

`pvalue=0.9999999999999999`

### If they were very different that `pvalue` would be less than 0.01, but its almost 100 times that. Meaning the makeup of King County as a whole is not significantly different from the makeup of food deserts in King County. They look pretty much the same because they are pretty much the same.

## Alright, so what about income? Are food deserts mostly made of lower income earners?

![Income Composition of Respondents in Food Deserts](/img/income_des.png)

### That is also pretty surprising. Or, perhaps you haven't heard, the median income in King County in 2015 was a cool $[75k](https://www.kingcounty.gov/independent/forecasting/King%20County%20Economic%20Indicators/Household%20Income.aspx)!

![Income Composition of All Respondents](/img/income_total.png)

##### Just for a little perspective, according to MIT, the average expenses for a family of four in King County are about $[80k](http://livingwage.mit.edu/counties/53033).

### Well, these look the same, too. Are they? Let's do the math things!

`pvalue=0.9999999999999997`

### That `pvalue` is, again, too high to find any difference between those in and those out of food deserts.

## We're just about out of space here, folks. Let's check out one more thing. This is where the rubber meets the road. Do people in King County food deserts eat fresh fruits and veggies less often?

![Consumption of Fruits and Veg](/img/consumption.png)

## Most eat fruits and vegetables every day. That's great news! 

### So, like, what's the deal with this food desert thing anyway? I mean, I wouldn't necessarily think that over 80% of people in food deserts would eat vegetables every day. It doesn't seem to make much of a difference. Is this food desert thing even a thing? In short, yes. If you need further convincing please check out the links at the top. King County isn't typical in many ways. Two obvious differences are those median income and cost of living stats, yikes. Another factor not in our data is that public transportation is extensive throughout the area and available door to door for many vulnerable populations. Online ordering and door to door delivery services are also abundant, as are farmer's markets. Also, biking and walking and are encouraged with infrastructure improvements and public outreach. 

### Access to affordable, healthy food options remains a challenge in many communities. Thankfully, getting fruits and vegetables in King County isn't too hard, even if you're over a mile from the store and live in a lower income neighborhood. 

### It would be interesting to do similar research in other communities around the country and see just how right I am in my assumption that the USDA is right about food deserts being generally distinctive from their surrounding food oases.
