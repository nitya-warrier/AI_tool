Google Search Summary

This program takes an input from the user, performs a Google search for the given topic, excludes certain topics as specified by the user, and provides a summarized report based on the search results.

Usage
To use this program, simply run the following command in your terminal:

python customized_report_summary.py

You will then be prompted to enter a topic to search for. Once you enter your desired topic and press Enter, the program will perform a Google search and display a list of search results.

You can then specify any topics that you wish to exclude from the search results. For example, if you are searching for information about dogs but want to exclude the links 2 and 3 because they are from the same website, you can do so when prompted.

After excluding any unwanted topics, the program will provide a summarized report of the remaining search results.

Dependencies
This program requires the following dependencies:

google (for performing the Google search)

beautifulsoup4 (for parsing the search results)

trafilatura (for scraping the text from the webpages)

nltk(for breaking the results into chunks for feeding the AI model)

openai

spacy

You can install these dependencies by running the following command:

pip install google
and so on for the other packages. 
Limitations
Please note that this program has some limitations:

It relies on the accuracy of Google search results, which may not always be reliable.
You can only give the search numbers to exclude. 
This will give you only the top 10 results. 
It may not be able to accurately summarize certain types of content, such as images or videos.
It may not work properly if your internet connection is slow or unstable.


Contact
If you have any questions or feedback about this program, please contact me at nitya.warrier@gmail.com 
