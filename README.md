The goal of this tool is to reduce the thousands of hours of individuals wasted by automating the tedious task of making sure evnrionmental project guidelines are regulation compliant, however this can be expanded to any industry. 
Most web based AI checkers introduce data privacy issues, and the goal of this framework is to provide a localized, protected tool for individuals to run on their own systems. This ensures privacy from cloud serivces. 

This is a prototype FRAMEWORK with the setup to allow individuals an easy localized AI base to develop further. 

The core tool coonsists of two functionalites:
- The first is a basic chatbot funcitoning on RAG enhanced ranking retreival in order to provide answers to questions related to the content provided to the tool.
- The second is the framework for a comprehenisve project guidelines checker against local regulations:
    - The tool acheives this by processing each section of the project guidelines, retrieving relevant regulations, and making sure that the project guidelines adhere to the local regulations.
    - "Adherence" includes JSON structures of the project guidelines and relevant regulations being directly compared for factual statements.
    - "Adherence" also includes AI analyzing complex statements not suitable for JSON to make sure that prject guidelines are compliant.
 
Potential Pitfalls: 
- One of the major pitfalls of relying on AI for complex statements is the quality of the AI itself. Small models (<1B parameters) struggle heavily with this type of work, sometimes producing incoherrent outputs.
- This means that the tool requires smarter, larger models operating at its core.
- The tool has been made to be able to parse general PDF structures, however it may struggle or fail with obsolte or niche PDF strucutures, causing failure during parsing.  

