var socket = io.connect('/liveuploader');

var Button = ReactBootstrap.Button;
var Popover = ReactBootstrap.Popover;
var Input = ReactBootstrap.Input;

var VideoCanvas = React.createClass({
    getInitialState() {
      return {
        visible : true,
        videoRotate180: false,
      };
    },
    propTypes: {
      width: React.PropTypes.number.isRequired,
      height: React.PropTypes.number.isRequired,
      uploadWidth: React.PropTypes.number.isRequired,
      uploadHeight: React.PropTypes.number.isRequired,
      captureInterval: React.PropTypes.number.isRequired,
      updateFrameSize: React.PropTypes.func.isRequired,
    },
    componentDidMount() {
      this.video = ReactDOM.findDOMNode(this.refs.video);
      // visible canvas
      this.canvas = ReactDOM.findDOMNode(this.refs.canvas);
      this.ctx = this.canvas.getContext('2d');
      // video cache canvas
      this.upload_canvas = ReactDOM.findDOMNode(this.refs.upload_canvas);
      this.upload_ctx = this.upload_canvas.getContext('2d');

      // initialize video
      var that = this;
      var p = navigator.mediaDevices.getUserMedia({audio: false, video: true});
      p.then(function(mediaStream) {
          that.video.src = window.URL.createObjectURL(mediaStream);
          that.video.onloadedmetadata = that.updateFrameSize;
          that.video.onresize = that.updateFrameSize;
          // start video capture
          that.play();
      });
    },
    updateFrameSize() {
      this.props.updateFrameSize(this.video.videoWidth, this.video.videoHeight);
    },
    drawCanvasFull(src) {
      this.ctx.drawImage(src, 0, 0, this.refs.canvas.width,
                                    this.refs.canvas.height);
    },
    drawHiddenCanvasFull(src) {
      this.upload_ctx.drawImage(src, 0, 0, this.refs.upload_canvas.width,
                                           this.refs.upload_canvas.height);
    },
    setVisibility(v) {
      this.setState({visible: v});
    },
    handleCheckboxRotate180(v) {
      this.setState({videoRotate180: !this.state.videoRotate180});
    },
    getVideoRotate180() {
      return this.state.videoRotate180;
    },
    play() {
      this.stop(); // escape double loop

      // draw loop function
      var that = this;
      var drawLoop = function() {
        // draw to canvases
        if (that.state.visible) that.drawCanvasFull(that.video);
        that.drawHiddenCanvasFull(that.video);
        // recursive call
        that.loopId = setTimeout(drawLoop, that.props.captureInterval);
      };
      drawLoop();  // initial call
    },
    stop() {
      if (this.loopId) {
        clearTimeout(this.loopId);
        this.loopId = null;
      }
    },
    render() {
      return (
        <div>
          <Button onClick={this.handleCheckboxRotate180}>Rotate</ Button>
          <video ref="video"
           style={{display: "none"}}
           width={this.props.width}
           height={this.props.height}
           autoPlay="1" />
          <canvas ref="canvas"
           className="img-responsive"
           style={{backgroundColor: 'black'},
                  {transform: this.state.videoRotate180 ? 'rotate(180deg)'
                                                        : 'none'},
                  {WebkitTransform: this.state.videoRotate180 ? 'rotate(180deg)'
                                                              : 'none'}}
           width={this.props.width}
           height={this.props.height} />
          <canvas ref="upload_canvas"
           style={{display: "none"}}
           width={this.props.uploadWidth}
           height={this.props.uploadHeight} />
        </div>
      );
    }
});


var VideoUI = React.createClass({
    propTypes: {
      videoWidth: React.PropTypes.number.isRequired,
      videoHeight: React.PropTypes.number.isRequired,
      uploadWidth: React.PropTypes.number.isRequired,
      uploadHeight: React.PropTypes.number.isRequired,
      captureInterval: React.PropTypes.number.isRequired,
      onReset: React.PropTypes.func,
    },
    getInitialState() {
      return {
        uploading: false,
      };
    },
    componentDidMount() {
    },
    startUploading() {
      this.setState({uploading: true});
      this.refs.videocanvas.setVisibility(false);
      // initial call
      this.uploadImg();
    },
    uploadImg() {
      // get image data
      var data = this.refs.videocanvas.upload_canvas.toDataURL('image/jpeg');
      // emit
      var rotate = this.refs.videocanvas.getVideoRotate180();
      socket.emit('upload_img', {img: data, rotate: rotate});
    },
    onResult(data) {
      // check whether this request is canceled
      if (!this.state.uploading) return;

      // draw received image
      var that = this;
      var img = new Image();
      img.src = data.img;
      img.onload = () => {
        that.refs.videocanvas.drawCanvasFull(img);
        // recursive call
        that.uploadImg();
      };
    },
    onReset() {
      // call parent event
      if (this.props.onReset) this.props.onReset();
      // reset states
      this.setState({uploading: false});
      this.refs.videocanvas.setVisibility(true);
      // restart video
      this.refs.videocanvas.play();
    },
    onVideoSizeChanged(width, height) {
      // set component size
      this.props.onVideoSizeChanged(width, height);
    },
    playVideo() {
      this.refs.videocanvas.play();
    },
    stopVideo() {
      this.refs.videocanvas.stop();
    },
    render() {
      return (
        <div>
          <VideoCanvas ref="videocanvas" 
           width={this.props.videoWidth}
           height={this.props.videoHeight}
           uploadWidth={this.props.uploadWidth}
           uploadHeight={this.props.uploadHeight}
           captureInterval={this.props.captureInterval}
           updateFrameSize={this.onVideoSizeChanged} />
          <div style={{maxWidth: this.props.videoWidth}}>
            <Button bsStyle="default" bsSize="large" className="col-xs-6"
             onClick={this.playVideo}>Play</Button>
            <Button bsStyle="default" bsSize="large" className="col-xs-6"
             onClick={this.stopVideo}>Stop</Button>
            <Button bsStyle="primary" bsSize="large" className="col-xs-6"
             disabled={this.state.uploading ? true : false}
             onClick={this.startUploading}>Start</Button>
            <Button bsStyle="danger" bsSize="large" className="col-xs-6"
             onClick={this.onReset}>Reset</Button>
          </div>
        </div>
      );
    }
});


var Message = React.createClass({
    propTypes: {
      width: React.PropTypes.number.isRequired,
    },
    getInitialState() {
      return {
        message: '',
      };
    },
    componentDidMount() {
    },
    setMessage(message) {
      this.setState({message: message});
    },
    render() {
      return (
        <div className="col-xs-12"
         style={{maxWidth: this.props.width}}>
          {(() => {
            if (this.state.message) {
              return (
                <Popover id={0} placement="bottom"
                 style={{maxWidth: this.props.width}}>
                  {this.state.message}
                </Popover>
              );
            }
          })()}
        </div>
      );
    }
});


var MainView = React.createClass({
    getInitialState() {
      return {
        videoWidth: 400,
        videoHeight: 300,
        uploadResol: 1.0,  // resolution scale
        captureInterval: 15,
      };
    },
    componentDidMount() {
      // socket.io event
      socket.on('connect', this.onConnect);
      socket.on('response', this.onResponse);
    },
    onConnect() {
    },
    onResponse(data) {
      if (data.img) this.refs.videoui.onResult(data);
      if (data.msg) this.refs.message.setMessage(data.msg);
    },
    onReset() {
      this.refs.message.setMessage('');
    },
    onVideoSizeChanged(width, height) {
      this.setState({videoWidth: width});
      this.setState({videoHeight: height});
    },
    render() {
      return (
        <div className="center-block">
          <div className="container">
            <div className="row">
              <div className="col-xs-12">
                <VideoUI ref="videoui"
                 videoWidth={this.state.videoWidth}
                 videoHeight={this.state.videoHeight}
                 uploadWidth={this.state.videoWidth * this.state.uploadResol}
                 uploadHeight={this.state.videoHeight * this.state.uploadResol}
                 captureInterval={this.state.captureInterval}
                 onReset={this.onReset}
                 onVideoSizeChanged={this.onVideoSizeChanged} />
              </div>
              <div className="col-xs-12">
                <Message ref="message"
                 width={this.state.videoWidth} />
              </div>
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
