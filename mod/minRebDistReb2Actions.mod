tuple Edge{
  int i;
  int j;
}

tuple edgeAttrTuple{
    int i;
    int j;
    int t;
}

tuple accTuple{
  int i;
  float n;
  float d;
  float a;
}

string path = ...;

{edgeAttrTuple} edgeAttr = ...;
{accTuple} accInitTuple = ...;

{Edge} edge = {<i,j>|<i,j,t> in edgeAttr};
{int} region = {i|<i,n,d,a> in accInitTuple};

float time[edge] = [<i,j>:t|<i,j,t> in edgeAttr]; // TODO: distance --> we have no distance (replace with time?)
float departure[region] = [i:d|<i,n,d,a> in accInitTuple]; // TODO: desiredVehicles
float vehicles[region] = [i:n|<i,n,d,a> in accInitTuple]; // TODO: vehicles
float arrival[region] = [i:a|<i,n,d,a> in accInitTuple]; // TODO: vehicles

dvar int+ demandFlow[edge];
dvar int+ rebFlow[edge];

minimize(sum(e in edge) (rebFlow[e]*time[e]));
subject to
{
  forall(i in region)
    {
    sum(e in edge: e.i==i && e.i!=e.j) rebFlow[e] == departure[i];
	sum(e in edge: e.j==i && e.i!=e.j) rebFlow[e] == arrival[i];
    }
}

main {
  thisOplModel.generate();
  cplex.solve();
  var ofile = new IloOplOutputFile(thisOplModel.path);
  ofile.write("flow=[")
  for(var e in thisOplModel.edge)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.j);
         ofile.write(",");
         ofile.write(thisOplModel.rebFlow[e]);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.close();
}