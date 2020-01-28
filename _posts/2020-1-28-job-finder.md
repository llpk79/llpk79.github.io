---
layout: default
title: Job Finder
subtitle: Scrape Indeed.com, vectorize job listings with Spacy and find job descriptions similar to resume with scikit-learn KNN.
image: /img/handshake_orig.jpg
---

## Job Finder
Write a job description for your ideal job, and use the power of NLP to search job sites for your ideal position so you can spend your time preparing for interviews instead of searching through listings!

Best results are expected by providing a job description writen in the style of a job description. The NLP algorithm is looking at document similarity as a whole, not simply searching for keywords or phrases.

Docker container runs on any machine. [What is Docker?](https://docs.docker.com/engine/docker-overview/)

After starting the program the command line prompts you to enter how many pages to scrape, how many jobs to return, where to search, what search phrase to use, and the name of your text file.

Currently Job-finder searches only Indeed.com.

Python program uses:
- BeautifulSoup4 to scrape Indeed.com for job descriptions
- The Spacy NLP library to compare your ideal job description (or resume) to job descriptions found on Indeed, and find and remove duplicate listings
- A Scikit-Learn unsupervised machine learning technique called Nearest Neighbors to find the jobs best matching the document you provide

For download and installation instructions see [Github](https://github.com/llpk79/Job-finder/packages/111889).