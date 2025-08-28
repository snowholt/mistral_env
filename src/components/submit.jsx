// This is the Node.js code for your Alibaba Cloud Function Compute function.
// It uses the @alicloud/pop-core package to interact with the DirectMail API.
// You need to install this dependency in your Function Compute environment.

const Core = require('@alicloud/pop-core').RPCClient;

// IMPORTANT: Replace with your actual credentials and region
// Store these in environment variables for security in a real-world app.
const accessKeyId = 'process.env.ALIBABA_ACCESS_KEY_ID || 'YOUR_ACCESS_KEY_ID'';
const accessKeySecret = 'process.env.ALIBABA_ACCESS_KEY_SECRET || 'YOUR_ACCESS_KEY_SECRET'';
const regionId = 'me-central-1a'; // e.g., 'ap-southeast-1' for Singapore

// Initialize the DirectMail client
const client = new Core({
  accessKeyId: accessKeyId,
  accessKeySecret: accessKeySecret,
  endpoint: `https://dm.${regionId}.aliyuncs.com`,
  apiVersion: '2015-11-23'
});

// The main handler function for Function Compute
exports.handler = async (event, context, callback) => {
  try {
    // Parse the incoming request body
    const requestBody = JSON.parse(event.toString());
    
    // Extract form data from the request body
    const { firstName, lastName, email, phonenumber, company, companySize, message } = requestBody;

    // IMPORTANT: Replace this with the email you want to send the form submissions TO
    const toAddress = 'info@gmail.sa';
    
    // IMPORTANT: Replace this with the sender address you configured in DirectMail
    const fromAddress = 'info@gmail.sa';
    const fromAlias = 'Contact Form';

    // Construct the email body with the form data
    const emailSubject = `New Contact Form Submission from ${firstName} ${lastName}`;
    const emailBody = `
      <h1>New Contact Form Submission</h1>
      <p><strong>Name:</strong> ${firstName} ${lastName}</p>
      <p><strong>Email:</strong> ${email}</p>
      <p><strong>Phone:</strong> ${phonenumber}</p>
      <p><strong>Company:</strong> ${company}</p>
      <p><strong>Company Size:</strong> ${companySize}</p>
      <p><strong>Message:</strong></p>
      <p>${message}</p>
    `;

    // Call the DirectMail API to send the email
    const params = {
      'RegionId': regionId,
      'AccountName': fromAddress,
      'FromAlias': fromAlias,
      'ReplyToAddress': true, // Allows you to reply directly to the user's email
      'ToAddress': toAddress,
      'Subject': emailSubject,
      'HtmlBody': emailBody,
    };

    const requestOption = {
      method: 'POST'
    };

    // Send the email and get the response
    const result = await client.request('SingleSendMail', params, requestOption);

    // Return a success response
    callback(null, {
      statusCode: 200,
      body: JSON.stringify({ message: "Message sent successfully!" }),
      headers: {
        "Content-Type": "application/json",
        // Allow CORS requests from your React app's domain
        "Access-Control-Allow-Origin": "*", // For development, you can lock this down later
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
      }
    });

  } catch (error) {
    console.error("Error processing form submission:", error);
    // Return an error response
    callback(null, {
      statusCode: 500,
      body: JSON.stringify({ message: "An error occurred." }),
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
      }
    });
  }
};
