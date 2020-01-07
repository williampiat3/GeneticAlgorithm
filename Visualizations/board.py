import argparse
import json
import pickle
from flask import jsonify,request,Flask,render_template, redirect, url_for


#Starting the flask app here to be able to define the routing
app = Flask(__name__)
#transform the pops object into the format for the D3 framework
#the format of the output is made to match the one of the d3 file that creates the sankey diagramm
def format_pops_in_json(path,index_fit):
    #loading population 
    with open(path,'rb') as pickle_file:
        hparams=pickle.load(pickle_file)

        pops=pickle.load(pickle_file)

    #this is the structure for the d3 js: nodes are the dots
    #links are the binding between two nodes
    json_struct={"population_size":len(pops[0]),
                 "nodes":[],
                 "links":[]



    }
    #index node is the index of an individual in the nodes array
    #it is mandatory for linking individuals
    index_node=0
    previous_gen={}
    for i,pop in enumerate(pops):
        next_adresses={}
        for indiv in pop:
            #for each indiv we are creating a node
            json_struct["nodes"].append({"generation":i,
                                         "name":indiv["id"],
                                         "formula":indiv["phen"],
                                         "fitness":indiv["fits"][int(index_fit)] if index_fit.isdigit() else indiv["phen"][index_fit] ,
                                         })
            next_adresses[indiv['id']]=indiv
            next_adresses[indiv['id']]["index_node"]=index_node
            #if the node has parents we link the node to its parents
            if indiv["parents"] != None:
                if indiv["mutated"] != None:
                    op=2
                else:
                    op=1
                for parent in indiv["parents"]:
                    json_struct["links"].append({"source":previous_gen[parent]["index_node"],
                                                 "target":index_node,
                                                 "op":op

                    })
                index_node+=1
                continue
            #If the individual is mutated we link it to its original individual
            if indiv["mutated"] != None and indiv["parents"] == None:
                json_struct["links"].append({"source":previous_gen[indiv["mutated"]]["index_node"],
                                                 "target":index_node,
                                                 "op":0

                    })
                index_node+=1
                continue
            # if the individual doesn't have parents, is not mutated and doesn't belong to the first generation
            # then it is some elitism and we link it to itself
            if indiv["mutated"] == None and indiv["parents"] == None and (not (previous_gen=={})):
                json_struct["links"].append({"source":previous_gen[indiv["id"]]["index_node"],
                                                 "target":index_node,
                                                 "op":-1

                    })
                index_node+=1
                continue
            index_node+=1
        
        previous_gen=next_adresses
    return json_struct

@app.route('/')
def index():
    return redirect(url_for('main_dashboard'))

@app.route('/dashboard')
def main_dashboard():
    index_fit = request.args.get('fitness')
    if index_fit == None:
        index_fit='0'
    json_str=format_pops_in_json(args.log_file,index_fit)
    return render_template('graph_d3.html', data_gen = json_str)



parser = argparse.ArgumentParser(description='Display generations')
parser.add_argument('--log_file',help='Log_file address')
args = parser.parse_args()

app.run(host= '0.0.0.0')



