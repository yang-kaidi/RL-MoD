tuple edge{
	int i;
	int j;
}


tuple dtuple{
	int i;
	int j;
	float t;
	float f;
	float v;
}

tuple ptuple{
	int i;
	int j;
	float t;
	float v;
}

tuple sedge{
	int s;
	int o;
	int d;
}
 
float gamma = ...;
string path = ...;
{sedge} solutionEdge = ...;
{dtuple} demandEdgeTuple = ...;
{ptuple} pickupEdgeTuple = ...;
{edge} demandEdge = {<i,j>|<i,j,t,f,v> in demandEdgeTuple};
{edge} pickupEdge = {<i,j>|<i,j,t,v> in pickupEdgeTuple};
float fare[demandEdge] = [<i,j>:f|<i,j,t,f,v> in demandEdgeTuple];
float demandFlow[demandEdge] = [<i,j>:v|<i,j,t,f,v> in demandEdgeTuple];
float pickupFlow[pickupEdge] = [<i,j>:v|<i,j,t,v> in pickupEdgeTuple];
float pcktt[pickupEdge] = [<i,j>:t|<i,j,t,v> in pickupEdgeTuple];
float paxtt[demandEdge] = [<i,j>:t|<i,j,t,f,v> in demandEdgeTuple];

{int} region = {o|<s,o,d> in solutionEdge};


dvar float+ flow[solutionEdge];
dvar float e1[demandEdge];
dvar float e2[pickupEdge];
maximize sum(e in solutionEdge) flow[e] * (fare[<e.o,e.d>] - gamma*pcktt[<e.s,e.o>] - gamma*paxtt[<e.o,e.d>]) - 
10000*sum(e in demandEdge )abs(e1[e])-10000*sum(e in pickupEdge )abs(e2[e]);
subject to{

		forall(e in demandEdge)
		  	sum(ee in solutionEdge: ee.o==e.i && e.j==ee.d) flow[ee] == demandFlow[e] +e1[e];
	
		forall(e in pickupEdge)
		  	pickupFlow[e]== sum(ee in solutionEdge: ee.o==e.j && e.i==ee.s) flow[ee] +e2[e]; 
		  	
}

main{
  var file = new IloOplOutputFile(thisOplModel.path);
  thisOplModel.generate();
	cplex.solve();
  	file.write("flow=[");
  for(var e in thisOplModel.solutionEdge)
	{
		file.write("(");
		file.write(e);
		file.write(",");
		file.write(thisOplModel.flow[e]);
		file.write(")");		
	
	}
	
	file.writeln("];");
}