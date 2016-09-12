var socket = io.connect('/viewer');

var Nav = ReactBootstrap.Nav;
var Navbar = ReactBootstrap.Navbar;
var NavItem = ReactBootstrap.NavItem;


var ImageView = React.createClass({
    propTypes: {
      data: React.PropTypes.object.isRequired,
    },
    render() {
      return (
        <div className="thumbnail"
         style={{marginBottom: '0px'}}>
          <img src={this.props.data.img || ""}
           style={{width: this.props.data.width || 'auto'}}
           className="img-responsive" >
            {(() => {
                  if (this.props.data.cap) {
                    return (
                      <div
                       style={{wordWrap: 'break-word',
                               maxWidth: this.props.data.width || 'auto'}}
                       className="text-center" >
                        {this.props.data.cap || ""}
                      </div>
                    );
                  }
            })()}
          </img>
        </div>
      );
    }
});


var ImageList = React.createClass({
    propTypes: {
      images: React.PropTypes.object.isRequired,
    },
    render() {
      return (
        <div>
          {Object.keys(this.props.images).sort().map((key) => {
                var data = this.props.images[key];
                if (data) {
                  return (
                    <div key={key}
                     className="pull-left"
                     /* className="col-lg-2 col-md-2 col-sm-3 col-xs-4" */>
                     <ImageView data={data} />
                    </div>
                  );
                }
          })}
        </div>
      );
    }
});


var TabNavigator = React.createClass({
    propTypes: {
      tabNames: React.PropTypes.array.isRequired,
      activeTab: React.PropTypes.any,
      changeActiveTab: React.PropTypes.func.isRequired,
    },
    handleSelect(key) {
      this.props.changeActiveTab(key);
    },
    render() {
      return (
      <Navbar inverse>
        <Nav bsStyle="tabs"
             activeKey={this.props.activeTab}
             onSelect={this.handleSelect}>
             {this.props.tabNames.sort().map((name, idx) => {
                if (name) {
                  return (
                    <NavItem key={name} eventKey={name}>{name}</NavItem>
                  );
                }
          })}
        </Nav>
      </Navbar>
      );
    }
});

var MainView = React.createClass({
    getInitialState() {
      return {
        tabNames: [],
        imagesCollection: {},
        activeTab: null,
      };
    },
    componentDidMount() {
      // socket.io event
      socket.on('connect', this.onConnect);
      socket.on('update', this.onUpdate);
    },
    onConnect() {
      // clear
      this.setState({tabNames: []});
      this.setState({imagesCollection: {}});
      // initial request
      socket.emit('update');
    },
    onUpdate(values) {
      var that = this;
      if (!values) return;
      values.forEach(function(v, i) {
          if (!v) return;  // empty
          var tab = v[0];
          var name = v[1];
          var data = v[2];
          // update tab
          if (name) { // create new tab (or do nothing)
            that.createTab(tab);
          } else { // remove tab
            that.removeTab(tab);
          }
          // update image
          that.state.imagesCollection[tab][name] = data; // set
          if (i == values.length - 1) { // update
            that.setState({imagesCollection: that.state.imagesCollection});
          }
      });
    },
    createTab(tab) {
      if (this.state.tabNames.indexOf(tab) < 0) {
        // tab names
        this.setState({tabNames: this.state.tabNames.concat(tab)});
        // tab images
        this.setState({
            imagesCollection: Object.assign(this.state.imagesCollection,
                                            {[tab]: {}})
        });
        // set default tab
        if (!this.state.activeTab) this.changeActiveTab(tab);
      }
    },
    removeTab(tab) {
      // tab names
      this.setState({tabNames: this.state.tabNames.filter(function(v) {
              return v != tab;
      })});
      // active tab
      if (this.state.activeTab === tab) {
        if (this.state.tabNames.length >= 0) {
          this.setState({activeTab: this.state.tabNames[0]});
        } else {
          this.setState({activeTab: null});
        }
      }
    },
    changeActiveTab(tab) {
      this.setState({activeTab: tab});
    },
    render() {
      return (
        <div>
          <TabNavigator ref="tab"
           tabNames={this.state.tabNames}
           activeTab={this.state.activeTab}
           changeActiveTab={this.changeActiveTab}/>
          <div className="container-fluid">
            <div className="center-block">
            {(() => {
                var images = this.state.imagesCollection[this.state.activeTab];
                if (images) {
                  return (<ImageList images={images} />);
                }
            })()}
            </div>
          </div>
        </div>
      );
    }
});


ReactDOM.render(
  <MainView />,
  document.getElementById('content')
);
