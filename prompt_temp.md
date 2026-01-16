Check the Backend and Website `_website_snapshot`.
- Add the `_website_snapshot` to the git to follow the changes. We need to update the gitignore to not ignore this folder, but ignore the dependencies/react modules inside it.

- I need to improve Admin dashboard and add these features: 
1. On Metric section, add the GPU benchmarking (token per sec) benchmarking. 
2. Update the Knowledge Base (RAG)/ AI Agent configuration for adding the instruction to LLM model for both Admin and Customer dashboard, and make it easy and wizard based to configure, letting user to select the language (English, Arabic, or Both), 
    - Consider these options for knowledge base: 
        - Upload PDF documents
        - Upload URL links
        - Upload text files
        - Upload Word documents
        - Upload Excel files
- On knowledge base: 
    • A space for brief of the business (what you do ? ).
    • Is your business retail or services or both? (option that user selects one of three)
Retail: table with items, description, prices(range), warranty and shipping cost if applicable.
Services: a table of services list, prices, description, service time number of  and warranty. (it should be done user friendly wizard/Table based).
Both: all of above >> Services / Retail
    • Website link? A space to add a website.
    • Do you have a business documents (flyer, pricelist, profile……)? A space to upload the documents. ((it should be done user friendly wizard/Table based))
    • Do you want to link your schedule to services to be able to make appointments?
Yes: a space to add a link.>>>To be added in AI agent based on sub (basic Advanced) It should be done user friendly wizard based.

    • Do you want to add your business locations? Yes: a table to add the branch name, location and working hours.

    • Do you want to add a contact number for each branches/extension? Yes: a table to add contact number and its value.


    • Do you have promotion? Yes: a space to add the promotion details.>> It should be done user friendly wizard/Table based. 

- Retail: table with items, description, prices(range), warranty and shipping cost if applicable.
Services: a table of services list, prices, description, service time number of  and warranty.


- The goal is to make the knowledge base and AI agent (basic and advanced)configuration as easy as possible for the user, with clear instructions and a step-by-step wizard to guide them through the process. (Also keep the current basic and advance raw instruction section for advanced users who want to input their own instructions directly, if we did not provide specific needs for them).