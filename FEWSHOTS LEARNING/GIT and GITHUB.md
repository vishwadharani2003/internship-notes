<span style="color:#00b0f0">repository is where all the changes are saved</span>
- <span style="color:#2d0fc2">terminal is used to manipulate file structure using the command</span>
- <span style="color:#2d0fc2">we will be saving the history of the folder that the git provides us</span> 
- repository means a folder-how to do it is by git inti command->we will be intializing a folder

- this git file will be hiddem in that folder in order to see the hiden folder we will be using 
![[Pasted image 20240120180706.png]]

- what is inside git we need to check using this ->ls .git

- ![[Pasted image 20240120180753.png]]
- touch CNN.txt-> is used to create CNN file in git
- git status ->is used get messages of the file names and the changes being amde
- ![[Pasted image 20240120181537.png]]
- in this untracked file the changes wont be visible->hostory is not saved
- how to make these changes visible
	- take a wedding scenario in that guest will be added to stage and then photo will be taken ->that's what happens here the untracked files will be added to the git history whose changes will be noted and tracked
	- in order to click a picture  of the guest ->like that we need to track down the history of the file we will give the command as->``git commit -m "CNN.txt file added" ``
	- already if there photo is taken ->by mistake we have added them to the stage we could use this code->``` git restore --staged CNN.txt ```
- in order to display what are the contents provided in the file file we need to use ->``` cat CNN.txt```
- 