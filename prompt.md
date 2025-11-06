### Issue:
- The Webrtc debug audio files are not being captured as expected.

### Input: 
- Debuging tool: `test_webrtc_simple.html`
- Beautyai-api.service journal log: `reports/logs/journal_backend_service.log`

### Task:
- Investigate the issue and provide a solution to ensure the Webrtc debug audio files are captured correctly. (in this phase we do nto need to connect to the LLM or Whistper model)
- Do nto create any new documents, just provide the solution here.
- Do not create new test scripts, just provide the solution here.


If something is unclear, ask me questions in **one single code block** using Markdown format as well.  
Make sure each question is:
- Clear and easy to understand  
- Includes a simple explanation (why you are asking it)  
- Provides examples where possible  
- Suggests possible answers if applicable  



- [ ] Step 1: Check preload configuration for whisper-large-v3-turbo
- [ ] Step 2: Add Whisper model loading to the debug capture endpoint
- [ ] Step 3: Add transcription during audio capture (after resampling to 16kHz)
- [ ] Step 4: Send transcription results to the browser via WebSocket or HTTP response
- [ ] Step 5: Update test_webrtc_simple.html to display transcription text
- [ ] Step 6: Test end-to-end: speak → capture → transcribe → display