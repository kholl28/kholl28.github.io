---
layout: page
title: Menstrual Period Predictor ML Model
description: with background image
img: assets/img/12.jpg
importance: 1
category: personal
related_publications: true
---

This project involved creating an accurate regression machine learning model that can be used to predict the start date of a person's next menstrual period (the target variable) based on their last three period dates, age, and BMI (the predictor variables).

The user's self-reported age and BMI will be used as secondary predictor variables that will adjust the final prediction outputted by the model.

Previous research has indicated that age and BMI has a statistically significant relationship with menstrual cycle length, in that women in different age groups and BMI categories experience significant differences in cycle length. Thus, the model will adjust its final prediction based on this User-inputted information.

This will be achieved through the use of a machine learning model that will be trained using data from a 2013 study by [Fehring et al.](https://epublications.marquette.edu/cgi/viewcontent.cgi?article=1002&context=data_nfp) titled "Randomized comparison of two internet-supported natural family planning methods". The dataset is freely-available to the public via Marquette University's e-Publications site (which can be accessed [here](https://epublications.marquette.edu/data_nfp/7/)) and is in .csv format.

This study collected data of the menstrual cycles of 159 anonymous American women over 12 months, with each woman having approximately 10 cycles logged. For each cycle, information such as the length of the cycle, the overall mean cycle length, estimated day of ovulation, length of menstruation, etc. is logged.

Although no model can predict with 100% accuracy the exact start date of someone's next period, identifying patterns among individual people in the lengths of each of their cycles would provide a person with more accurate, informed information, reducing uncertainty and allowing people to be more prepared.

{% raw %}

```html
<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
```

{% endraw %}
