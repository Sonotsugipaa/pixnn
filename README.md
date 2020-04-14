<h1> Pixnn </h1>

<h2> An Amatorial Neural Network Playground </h2>

<p>
	Pixnn is an OpenGL application that allows the user to draw shapes
	on a grid, or canvas, by adding blue or red points:
	a neural network will do all the dirty work and draw a
	<i>light blue / orange</i> image based on your inputs... or orange
	spikes, depending on its mood.
</p>

<h2> Mouse And Key Mappings </h2>

<table>
	<tr>
		<th> Key / Button </th>
		<th> Action </th>
	</tr>
	<tr> <td>Left Mouse Button</td>   <td>Add Red Point</td>          </tr>
	<tr> <td>Right Mouse Button</td>  <td>Add Blue Point</td>         </tr>
	<tr> <td>Middle Mouse Button</td> <td>Add White Point</td>        </tr>
	<tr> <td>SHIFT</td>               <td>Half-Strength Modifier</td> </tr>
	<tr> <td>PAGE-UP</td>             <td>Increase Learning Rate</td> </tr>
	<tr> <td>PAGE-DOWN</td>           <td>Decrease Learning Rate</td> </tr>
	<tr> <td>HOME</td>                <td>Increase Granularity</td>   </tr>
	<tr> <td>END</td>                 <td>Decrease Granularity</td>   </tr>
	<tr> <td>DELETE</td>              <td>Undo last point</td>        </tr>
	<tr> <td>Q / ESC</td>             <td>Quit</td>                   </tr>
	<tr> <td>R</td>                   <td>Reset AI</td>               </tr>
	<tr> <td>G</td>                   <td>Reset Canvas</td>           </tr>
	<tr> <td>T</td>                   <td>Toggle Show Canvas</td>     </tr>
	<tr> <td>D</td>                   <td>Toggle Derivatives</td>     </tr>
</table>

The mouse buttons can be held, in order to continuously add points; <br/>
beware, adding multiple points (or holding a mouse button without moving
the cursor) will place duplicate training samples and will affect the results -
more samples in a small zone means more training cycles will be performed there,
and less training cycles will be performed elsewhere.

<h2> Requirements and Dependencies </h2>

<p>
	Pixnn is built around OpenGL 1.5: compiling it and running it
	require the <code>GLFW3</code> and <code>GLEW</code> libraries
	(either static or dynamic), <code>GNU Make</code> and the C++
	compiler (and its STL) from GCC, a.k.a. <code>g++</code>.
</p>

The following packages should provide the necessary libraries:
- Pacman (Arch Linux)
	- gcc
	- make
	- glew
	- glfw-x11
- dpkg (Debian)</li>
	- g++ *(GCC version must be > 7)*
	- make
	- libglfw3
	- libglew-dev
	- libglew2.1

<h2> Compiling and Linking </h2>

<p>
	The application's <code>make</code> target is <code>bin/nncli</code>.
</p> <p>
	Additionally, the makefile provides a phony target,
	<code>reset</code>, to remove all compiled or linked binaries:
	it's a shortcut for deleting all files in <code>bin/</code>,
	all <code>.o</code> files under <code>build/*</code> and
	all <code>.a</code> files in <code>lib/</code>.
</p>

<h2> Notes </h2>

<ul>
	<li> The neural network is neither multi-threaded nor throttled:
	     it will use all the CPU time it can get from a single thread
	     for learning, and an additional throttled thread to compute
	     each pixel of the canvas with a frequency of 60 Hz. </li>
	<li> OpenGL and the C++ STL make memory-profiling difficult with
	     <code>valgrind</code>, therefore some memory leaks <i>may</i>
	     exist within the main application. </li>
</ul>
