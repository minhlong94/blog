---
toc: true
use_math: true
layout: post
description: No wording, stright to the problem
categories: [tricks]
title: "Ways to sync your code from PyCharm to a remote Linux server"
---

# Introduction
I am not a fan of long wording so let's jump straight into the problem. 

Have: a PyCharm project, a remote Linux server (SSH)

Want: sync the project to Linux

There are three ways I know of. Let's go straight to the one I recommend and you can read the rest later.

# Use PyCharm's Deployment
At the time of writing, I am using PyCharm Pro version 2021.2.3. I am not sure if this function is available in other versions.
The tutorial is adapted from: https://blog.csdn.net/zhaihaifei/article/details/53691873

1. Select `Tools => Deployment => Configuration`
2. Press the `+` icon to add new Deployment.
3. Choose `SFTF`. (I don't know what this is though)
4. Choose a name for your deployment. Any name.
5. Select the three dots thing on the right hand side of the `SSH Configuration` tab.
6. Click the `+` icon to add new SSH configuration. Then input your host, username and password there. The SSH is usually given in the format `username@host`. Remember to test connection. Then select `OK` to apply the connection.
7. In the currently opening window, select `Mapping` on the top middle, then change the `Deployment path` to the path you want to upload. The path will be `cd`-ed to. The deployment will behave as follows: Server login => `cd` to this path => Sync code.
8. Click `OK` to close the window.
9. (Optional) To sync the code on `Ctrl+S` click, go to `Tools => Options => Upload changed files to...`. Select this as `On explicit save action (Ctrl+S)`.    
10. On the Project tree on the right hand side of PyCharm, click the whole project's folder (or just the folder/file you want), then select `Tools => Deployment => Upload to <yourServerName>`. The code will be uploaded to the server.

And that's it for this method. Fastest to configure in my opinion.

# Use GitHub
I do not recommend this method. The method is to upload the project to GitHub, clone it to the server then `git pull` every time the code is updated. It is the easiest but will consume a lot of your time.

# Use JetBrains Gateway
Link: https://www.jetbrains.com/help/pycharm/remote-development-a.html. Depends on you. Method #1 is sufficient to me.